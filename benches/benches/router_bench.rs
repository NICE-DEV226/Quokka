//! Bench du routeur — DoD §8.3 : latence de classification < 2 ms.
//!
//! Le corpus mélange les trois chemins : court-circuit lexicaux
//! (overrides, code, prompt court) et le classifieur sémantique
//! (Stage 2), qui est le seul costing une allocation de vecteur.

use criterion::{Criterion, criterion_group, criterion_main};
use quokka_router::{RouteContext, Router};

/// Corpus représentatif : trivial, code, sémantique ambre.
const PROMPTS: &[&str] = &[
    "hi",
    "what is 2+2",
    "tell me a joke",
    "write a rust function that parses json",
    "fix this segmentation fault in my c code",
    "hello how are you today",
    "summarize this article in three sentences",
    "translate hello world to spanish",
    "debug this stack trace borrow checker error",
    "refactor this module architecture",
    "I need a careful architectural review of this concurrent rust service: \
     it uses tokio tasks, a shared cache and a postgres pool, and under load \
     the latency spikes. Where should I start debugging?",
    "I am planning a weekend trip to the mountains with my family, can you \
     suggest some easy hiking trails and a cozy mountain cabin to stay in?",
    "explain the mathematical derivation behind gradient descent optimizer \
     and how learning rate affects convergence in practice",
    "recommend a good movie for a rainy sunday night",
    "design a distributed system that handles millions of concurrent requests \
     with strong consistency guarantees across regions",
    "give me a recipe for pasta carbonara",
    "prove by induction that this recursive algorithm always terminates",
];

fn ctx() -> RouteContext {
    // Pas de garde mémoire en bench : on mesure le classifieur pur.
    RouteContext {
        available_ram_mb: Some(1 << 20),
        expert_size_mb: 0,
    }
}

fn bench_router(c: &mut Criterion) {
    let router = Router::new();
    let ctx = ctx();

    let mut group = c.benchmark_group("router");
    group.sample_size(200);
    for prompt in PROMPTS {
        group.bench_function(format!("route/{prompt:.40}"), |b| {
            b.iter(|| router.route(prompt, &ctx));
        });
    }
    // Et le pire cas réel : batch de 100 prompts hétérogènes.
    group.bench_function("route/corpus_100", |b| {
        b.iter(|| {
            for prompt in PROMPTS.iter().cycle().take(100) {
                let _ = router.route(prompt, &ctx);
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_router);
criterion_main!(benches);
