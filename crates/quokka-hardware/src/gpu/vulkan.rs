//! Détection Vulkan des GPU discrets via `ash`.
//!
//! Ce module est le seul endroit hors `quokka-core` qui parle à une ABI C
//! brute (le loader Vulkan). Chaque bloc `unsafe` porte son invariant
//! `// SAFETY:`, conformément à la règle mémoire du dépôt (DoD §8.4).
//!
//! Seuls les périphériques `DISCRETE_GPU` comptent : les iGPU exposent un
//! heap DEVICE_LOCAL découpé dans la RAM système, qu'il ne faut pas
//! rapporter comme VRAM dédiée (sinon la matrice de tiers promeut des
//! laptops à iGPU en Workstation).

use ash::{vk, Entry};

/// Meilleur GPU discret trouvé (plus grand heap DEVICE_LOCAL) + nombre.
#[derive(Debug, Clone)]
pub struct VulkanGpu {
    pub name: String,
    pub vram_bytes: u64,
    pub device_count: u32,
}

/// Les heaps plus petits sont des carve-outs iGPU de la RAM système,
/// pas de la VRAM dédiée.
const MIN_DEDICATED_HEAP_BYTES: vk::DeviceSize = 512 * 1024 * 1024;

/// Détecte les GPU discrets via l'ICD Vulkan. Renvoie `None` si le loader
/// (`libvulkan.so.1` / `vulkan-1.dll`) est absent ou sans GPU discret.
pub fn detect() -> Option<VulkanGpu> {
    // Chargement dynamique du loader : échec propre (pas de panic) sans ICD.
    // SAFETY: `Entry::load()` ne fait que résoudre les symboles Vulkan
    // (`vkGetInstanceProcAddr` et consorts) ; aucun objet Vulkan n'est créé
    // ici et l'échec retourne `Err` proprement.
    let entry = match unsafe { Entry::load() } {
        Ok(entry) => entry,
        Err(err) => {
            tracing::debug!("Vulkan loader unavailable ({err}), skipping Vulkan detection");
            return None;
        }
    };

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"quokka-hardware")
        .api_version(vk::make_api_version(0, 1, 0, 0));
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    // SAFETY: `create_info` ne baptise que `app_info`, qui survit à l'appel.
    // L'instance est détruite sur tous les chemins de sortie qui suivent.
    let instance = unsafe { entry.create_instance(&create_info, None) }.ok()?;

    // SAFETY: `instance` est vivante et exclusive à cet appel ; le contrat
    // interne de `scan_discrete_devices` est documenté sur sa déclaration.
    let result = unsafe { scan_discrete_devices(&instance) };

    // SAFETY: aucun objet Vulkan (device, mémoire, …) n'a été créé depuis
    // `instance` et plus rien ne l'emprunte après le retour de
    // `scan_discrete_devices` — la détruire ici est légal selon la spec.
    unsafe { instance.destroy_instance(None) };

    result
}

/// Énumère les physical devices et conserve le GPU discret au plus gros heap
/// DEVICE_LOCAL.
///
/// # Safety (contrat appelant)
///
/// `instance` doit rester vivante pendant tout l'appel et ne doit pas être
/// détruite concurremment. Aucun objet périphérique n'est créé ici.
unsafe fn scan_discrete_devices(instance: &ash::Instance) -> Option<VulkanGpu> {
    // SAFETY: simple requête d'énumération sur une instance valide ; la spec
    // garantit une liste complète des périphériques et le handle reste valide.
    let physical_devices = unsafe { instance.enumerate_physical_devices() }.ok()?;

    let mut best: Option<VulkanGpu> = None;
    let mut device_count: u32 = 0;

    for device in physical_devices {
        // SAFETY: requête de propriétés en lecture seule sur un handle
        // retourné par l'énumération ci-dessus ; aucune écriture aliasée.
        let properties = unsafe { instance.get_physical_device_properties(device) };

        if properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
            continue;
        }

        // SAFETY: requête de propriétés mémoire en lecture seule, même
        // validité de handle.
        let memory = unsafe { instance.get_physical_device_memory_properties(device) };

        let vram_bytes = memory
            .memory_heaps
            .iter()
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .filter(|heap| heap.size >= MIN_DEDICATED_HEAP_BYTES)
            .map(|heap| heap.size)
            .max()
            .unwrap_or(0);

        device_count += 1;
        if vram_bytes == 0 {
            tracing::debug!("discrete Vulkan device without usable local heap skipped");
            continue;
        }

        if best.as_ref().is_none_or(|gpu| vram_bytes > gpu.vram_bytes) {
            best = Some(VulkanGpu {
                name: decode_device_name(&properties.device_name),
                vram_bytes,
                device_count: 0, // corrigé après la boucle
            });
        }
    }

    best.map(|mut gpu| {
        gpu.device_count = device_count;
        gpu
    })
}

/// Décode le nom de périphérique fixe `char[256]` de Vulkan en `String`.
fn decode_device_name(raw: &[std::ffi::c_char]) -> String {
    let bytes = raw
        .iter()
        .take_while(|&c| *c != 0)
        .map(|&c| c as u8)
        .collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).into_owned()
}
