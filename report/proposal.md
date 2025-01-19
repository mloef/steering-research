## Background 

In the alignment and interpretability space, there are two prominent fields of thought: Mechanistic Interpretability (mech interp) and Representation Engineering (repeng). The main difference between the two is their philosophical approach. Mech interp aims to understand things at the neuronal, circuit level (bottoms up) and repeng aims to understand things from the aggregate, activation level (top down). Though the original repeng [paper](https://arxiv.org/abs/2310.01405) does a nice job drawing clear lines between the two approaches, in practice it's much less clear. 

Currently, mech interp is emerging as the leading method for understanding how LLMs work. While many of its advances have been applied to safety and alignment, little work has focused on using those advances to improve LLM capabilities. Repeng has used steering vectors from the raw LLM activations to promote concepts or ways of thinking in an LLM, but are highly manual to extract and apply, reduce output coherence, and are limited to simple ideas. Mech interp can also be used to extract and understand steering vectors, so this intersection is a promising area for apply mech interp to improve capabilities.

The mech interp space has been almost singularly focused on applying Sparse Autoencoders to decompress the activation space of the residual stream into a much larger and cleaner feature space that can be studied to understand what features appear in a model, and when they activate. However, this can also be used to extract steering vectors for one or more features that can then be applied to the original activations, resulting in steered generations. Framed that way, this holds the potential to improve LLM capabilities in a repeng manner.

This proposal will outline a plan for borrowing SAE feature extraction from mech interp and applying it in a repeng manner to do the following: 
1. automate the process of extracting steering vectors 
2. improve steered output coherence
3. develop steering & coherence evals 
4. explore more advanced steering applications & concepts


##  Planned Work

1. Automate vector creation
	1. Automate dataset generation for the positive and neutral prompts used to generate the vector
	2. Create algorithm for selecting strength coefficient when applying the vector
2. Improve LLM coherence while steered
	1. Experiment with SAEs and dimensionality reduction (PCA, UMAP, topk SAE feature activations)
	2. Optimize existing parameters: prompt, dataset size, dataset composition, layer targeting
	3. Experiment with other effects on coherence: base vs instruct model (with or without SAEs), different steering subject matter, SAE feature steering vs trained vector steering, new ways of steering beyond just "add a value to the activations" 
3. Develop evals, metrics, & human rating system for coherence and steerability
	1. Explore usefulness of metrics (perplexity vs base model, cosine similarity of embeddings against base steering prompt, search literature for instruction following metrics)
	2. Develop LLM-graded evals
	3. Experiment with human-in-the-loop evals: maybe develop a steerable chat interface and collect feedback on coherence & steerability?
4. (Stretch) Novel Applications
	1. Upscale to larger llama models
	2. Explore advanced applications: in-context learning, detailed characters, etc.


## Intersections with Plastic Labs

Interacting with language models at the *representation* level promises more granular control of generations. It's also the frontier of LLM interaction modes -- in a way, subverting the need to compress weights into output tokens allows language models to communicate in a way that's more native to them. It's in our interest to support research in this space to improve the way LLMs can represent *us*. 

If successful, we should have a much better understanding of working with representations and it should allow us to experiment in numerous novel directions.

### Intellectual respect

As foundation models continue to improve, the amount of intellectual respect they deserve increases as well. To us, this means relinquishing more cognitive tasks to the language model itself, rather than assuming we know what's best and hand-crafting some workflow. This theme presents a few directions of interest:

**Allowing the model to control its own steering**: if we provide the right meta-framework for the LLM to control its own steering, it might be able to apply its theory of mind skills in ways we couldn't think of (novelty, efficiency, etc).

**Allowing the model to swap its feature activations**: this goes along with some more abstract ideas, but if we operate at the representation level there's no reason those representations need to have directly interpretable meaning to humans -- they should be general enough to be applied in different contexts and steered in interesting ways dependent on the setting. The idea here is to give the model the ability to swap / manage the application of its feature activations on top of choosing which ones to derive.

**Distillation; using a large model to steer a small model**: if we could allow a larger, "smarter" model to observe the behavior of a smaller model and iterate on how best to steer it, then we end up with a teacher-student relationship. If the set of vectors needed to steer the smaller model approaches some number and plateaus, then this could be a very economical way to achieve individual alignment in future versions of Honcho.

### Personalization

Related to the above, if we can create the right meta-framework to allow the model to steer itself, we might be able to use that to create **user vectors** that represent some latent theory of mind about the user. It would be really interesting to experiment with what kind of representations could be derived/applied in abstract ways against the objective of predicting user behavior. Post-hoc analyses could be done to analyze what the vector represents.

### (Side Quest) Quantitative Memetics

At the highest level, representation vectors can be thought of as the quantification of **memes** in the original definition of the word: as ideas (broadly). If you view language models as the compressed representation of all human knowledge, then extracting specific representation vectors could be seen as a good proxy for how humans think of something. Being able to do math on these representations allows us to begin to explore in a more general sense how these ideas are formed, applied in different contexts, spread throughout culture, mutate over time, and more. Claude has already come up with an interesting [start](https://yousim.ai/share?code=gAAAAABmssNsxtql2ancIEXe0Te8n8kZs2gU9Htw8berSiGCnYi6TfGUQ0pSVUQeCozYA_qIxq4NMSLM8T032RGG5odiZAJv5tKgflj0QA9OLb2wbv7AdGJSzz1B_R2aP5HBfCCTYObOuOB21YUQkaiQqdowtUTQYiQPKSjI_LPgBUG3q388_VM=) to this textbook, but we think this could be a really interesting open-source contribution. Equally valid is the idea that our identities are just collections of memes. In this way, you could view Honcho as a grand experiment in quantitative memetics... where the meme we're trying to quantify is *you*. 

## Timeline and Capacity

- 20-30 hrs/week for 3 months (note: vacation 9/25-10/2, working from Asia timezone throughout)
- Month 1: One click vector creation from a prompt
- Months 2-3: Develop evals for coherence and steerability, then try to use them to refine steering coherence until it is usable on a wide range of steering goals without notable performance issues (<10% off base model on original benchmarks)
- Stretch goals as time permits

## Resources Required

- Tinybox access for llama 3 8B & Eleuther SAE experiments
- Occasional cloud/API credits for burst compute (<$1000 estimated total)

