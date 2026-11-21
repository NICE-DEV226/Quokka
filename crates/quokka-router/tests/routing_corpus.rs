//! Corpus de routage — gate de vérification Phase 3 (ROADMAP §3) :
//! ≥100 prompts labellisés, précision mesurée et rapportée.
//!
//! Le seuil exige 80 % : en dessous, le classifieur lexical rate trop
//! et le Stage 2 ONNX de l'ADR-0002 redevient nécessaire.

use quokka_router::{RouteContext, RouteTarget, Router};

fn ctx() -> RouteContext {
    RouteContext {
        available_ram_mb: Some(1 << 20),
        expert_size_mb: 0,
    }
}

/// (prompt, cible attendue). `F` = Flash, `E` = Expert.
const CORPUS: &[(&str, RouteTarget)] = &[
    // ---- Flash : conversation & trivial ----
    ("hi", RouteTarget::Flash),
    ("hello", RouteTarget::Flash),
    ("hey there", RouteTarget::Flash),
    ("good morning", RouteTarget::Flash),
    ("how are you", RouteTarget::Flash),
    ("what is 2+2", RouteTarget::Flash),
    ("tell me a joke", RouteTarget::Flash),
    ("thanks!", RouteTarget::Flash),
    ("bye", RouteTarget::Flash),
    ("what time is it", RouteTarget::Flash),
    ("what is the capital of france", RouteTarget::Flash),
    ("who wrote hamlet", RouteTarget::Flash),
    ("what does serendipity mean", RouteTarget::Flash),
    ("how do you spell necessary", RouteTarget::Flash),
    ("what is the weather today", RouteTarget::Flash),
    ("recommend a good book", RouteTarget::Flash),
    ("suggest a movie for tonight", RouteTarget::Flash),
    ("give me a recipe for pancakes", RouteTarget::Flash),
    ("how do I boil an egg", RouteTarget::Flash),
    ("what plants are easy to grow", RouteTarget::Flash),
    // ---- Flash : rédaction & reformulation ----
    ("rewrite this sentence more politely: give me the report now", RouteTarget::Flash),
    ("summarize this paragraph in one line please", RouteTarget::Flash),
    ("translate hello world to spanish", RouteTarget::Flash),
    ("translate this sentence into french for me", RouteTarget::Flash),
    ("write a short poem about the ocean", RouteTarget::Flash),
    ("write a haiku about autumn rain", RouteTarget::Flash),
    ("write a friendly email asking for a day off", RouteTarget::Flash),
    ("write a birthday message for my sister", RouteTarget::Flash),
    ("write a thank you note to my teacher", RouteTarget::Flash),
    ("draft a simple cover letter for a job application", RouteTarget::Flash),
    ("improve this paragraph so it sounds more professional", RouteTarget::Flash),
    ("make this text shorter without losing meaning", RouteTarget::Flash),
    ("turn these bullet points into a paragraph", RouteTarget::Flash),
    ("what should I say in a wedding toast", RouteTarget::Flash),
    ("give me a caption for this photo of my dog", RouteTarget::Flash),
    ("write an instagram bio for a coffee shop", RouteTarget::Flash),
    // ---- Flash : conseils & plans simples ----
    ("plan a weekend trip to the mountains", RouteTarget::Flash),
    ("suggest a cozy cabin for a family vacation", RouteTarget::Flash),
    ("what should I pack for a beach holiday", RouteTarget::Flash),
    ("give me ideas for a birthday gift for mom", RouteTarget::Flash),
    ("how do I train a puppy to sit", RouteTarget::Flash),
    ("what are easy home workouts", RouteTarget::Flash),
    ("give me tips for better sleep", RouteTarget::Flash),
    ("how do I remove a stain from a shirt", RouteTarget::Flash),
    ("suggest a menu for dinner party for six", RouteTarget::Flash),
    ("how do I learn basic photography", RouteTarget::Flash),
    ("what are the best beaches in portugal", RouteTarget::Flash),
    ("recommend a good science fiction book", RouteTarget::Flash),
    ("what laptop should I buy under 1000 euros", RouteTarget::Flash),
    ("help me pick a name for my new puppy", RouteTarget::Flash),
    ("what is a good gift for a colleague", RouteTarget::Flash),
    ("how do I start composting at home", RouteTarget::Flash),
    ("suggest a workout routine for beginners", RouteTarget::Flash),
    ("what are fun activities for kids on rainy days", RouteTarget::Flash),
    ("how do I make cold brew coffee", RouteTarget::Flash),
    ("give me a simple stretching routine", RouteTarget::Flash),
    // ---- Flash : questions de culture générale ----
    ("explain what a black hole is simply", RouteTarget::Flash),
    ("what causes thunder and lightning", RouteTarget::Flash),
    ("how do airplanes stay in the air", RouteTarget::Flash),
    ("what is the difference between weather and climate", RouteTarget::Flash),
    ("why is the sky blue", RouteTarget::Flash),
    ("what is photosynthesis in simple terms", RouteTarget::Flash),
    ("how does a microwave oven work", RouteTarget::Flash),
    ("what are the main causes of world war one", RouteTarget::Flash),
    ("who was the first person on the moon", RouteTarget::Flash),
    ("what is the capital of australia", RouteTarget::Flash),
    ("how many continents are there", RouteTarget::Flash),
    ("what is the largest ocean on earth", RouteTarget::Flash),
    ("when did the roman empire fall", RouteTarget::Flash),
    ("what is inflation in economics", RouteTarget::Flash),
    ("explain supply and demand like im five", RouteTarget::Flash),
    ("what is machine learning in simple words", RouteTarget::Flash),
    ("how do vaccines work at a high level", RouteTarget::Flash),
    ("what is the meaning of life according to philosophers", RouteTarget::Flash),
    ("explain quantum computing simply for a child", RouteTarget::Flash),
    // ---- Expert : code & debug ----
    ("write a rust function that parses json and returns a struct", RouteTarget::Expert),
    ("fix this borrow checker error in my async code", RouteTarget::Expert),
    ("debug this segmentation fault in my c program", RouteTarget::Expert),
    ("implement a binary search tree with insert and delete", RouteTarget::Expert),
    ("write a sql query joining users orders and products tables", RouteTarget::Expert),
    ("refactor this function to reduce cyclomatic complexity", RouteTarget::Expert),
    ("explain this stack trace from my rust program", RouteTarget::Expert),
    ("convert this python script to idiomatic rust", RouteTarget::Expert),
    ("write a regex that validates email addresses correctly", RouteTarget::Expert),
    ("implement a lru cache in python with o(1) operations", RouteTarget::Expert),
    ("fix the memory leak in this c++ vector usage", RouteTarget::Expert),
    ("write unit tests for this authentication module", RouteTarget::Expert),
    ("debug why this docker container keeps restarting", RouteTarget::Expert),
    ("optimize this nested loop from o(n^2) to o(n)", RouteTarget::Expert),
    ("implement a producer consumer queue with backpressure", RouteTarget::Expert),
    ("fix the race condition in this multithreaded code", RouteTarget::Expert),
    ("write a git hook that runs tests before commit", RouteTarget::Expert),
    ("explain the difference between mutex and rwlock in rust", RouteTarget::Expert),
    ("build a cli tool in rust that parses yaml config files", RouteTarget::Expert),
    ("fix this typescript type error in my react component", RouteTarget::Expert),
    // ---- Expert : architecture & systèmes ----
    ("design a distributed cache with consistent hashing", RouteTarget::Expert),
    ("review this microservices architecture for bottlenecks", RouteTarget::Expert),
    ("how should I shard a postgres database for scale", RouteTarget::Expert),
    ("design a rate limiter for a public api", RouteTarget::Expert),
    ("compare kafka vs rabbitmq for event streaming use case", RouteTarget::Expert),
    ("explain how cdn edge caching works architecturally", RouteTarget::Expert),
    ("design a notification system that must be exactly once", RouteTarget::Expert),
    ("how do i implement sagas for distributed transactions", RouteTarget::Expert),
    ("analyze the tradeoffs of monolith vs microservices for a startup", RouteTarget::Expert),
    ("design a multi region failover strategy for postgres", RouteTarget::Expert),
    ("explain consensus algorithms like raft in detail", RouteTarget::Expert),
    ("how does b trees indexing work in databases", RouteTarget::Expert),
    ("design an observability stack for a kubernetes cluster", RouteTarget::Expert),
    ("review this system design for a url shortener at scale", RouteTarget::Expert),
    ("explain how gpt attention mechanism computes q k v projections", RouteTarget::Expert),
    ("design a feature store for real time ml inference", RouteTarget::Expert),
    // ---- Expert : maths & raisonnement ----
    ("prove by induction that this algorithm terminates", RouteTarget::Expert),
    ("derive the closed form solution for this recurrence relation", RouteTarget::Expert),
    ("analyze the time complexity of this divide and conquer algorithm", RouteTarget::Expert),
    ("solve this integral using integration by parts step by step", RouteTarget::Expert),
    ("prove that the square root of 2 is irrational", RouteTarget::Expert),
    ("calculate the eigenvalues of this 3x3 matrix", RouteTarget::Expert),
    ("explain the proof of the pythagorean theorem", RouteTarget::Expert),
    ("solve this logic puzzle: three switches three bulbs", RouteTarget::Expert),
    ("derive bayes theorem from scratch with an example", RouteTarget::Expert),
    ("analyze the convergence of this gradient descent variant", RouteTarget::Expert),
    ("prove that there are infinitely many prime numbers", RouteTarget::Expert),
    ("compute the expected value of this probability distribution", RouteTarget::Expert),
    ("explain big o notation and amortized analysis with examples", RouteTarget::Expert),
    ("solve this combinatorics problem about committee selection", RouteTarget::Expert),
    ("derive the kinematics equations for projectile motion", RouteTarget::Expert),
    ("analyze the bias variance tradeoff in machine learning models", RouteTarget::Expert),
    // ---- Expert : analyse complexe & sécurité ----
    ("review this code for security vulnerabilities and injection risks", RouteTarget::Expert),
    ("explain the difference between symmetric and asymmetric encryption", RouteTarget::Expert),
    ("analyze this log file for signs of an intrusion attempt", RouteTarget::Expert),
    ("explain how tls handshake works at the cryptographic level", RouteTarget::Expert),
    ("compare these two sorting algorithms asymptotically", RouteTarget::Expert),
    ("explain transformer attention heads and positional encodings", RouteTarget::Expert),
    ("analyze the time complexity tradeoffs of different data structures", RouteTarget::Expert),
    ("explain how oauth2 authorization code flow works", RouteTarget::Expert),
    ("review this api design for rest vs graphql tradeoffs", RouteTarget::Expert),
    ("explain the difference between eventual and strong consistency", RouteTarget::Expert),
];

#[test]
fn routing_corpus_accuracy() {
    let router = Router::new();
    let ctx = ctx();
    let mut correct = 0_usize;
    let mut misses: Vec<(&str, RouteTarget, RouteTarget)> = Vec::new();
    let mut reasons: std::collections::HashMap<String, usize> = Default::default();

    for &(prompt, expected) in CORPUS {
        let d = router.route(prompt, &ctx);
        *reasons.entry(format!("{:?}", d.reason)).or_default() += 1;
        if d.target == expected {
            correct += 1;
        } else {
            misses.push((prompt, expected, d.target));
        }
    }

    let accuracy = correct as f64 / CORPUS.len() as f64;
    eprintln!("routing corpus: {}/{} = {:.1}%", correct, CORPUS.len(), accuracy * 100.0);
    eprintln!("reasons: {reasons:?}");
    for (p, e, g) in &misses {
        eprintln!("  MISS expected={e:?} got={g:?} :: {p}");
    }
    assert!(
        accuracy >= 0.80,
        "routing accuracy {accuracy:.1}% below 80% gate — {correct}/{}",
        CORPUS.len()
    );
}
