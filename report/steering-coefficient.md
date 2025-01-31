# Automatically Choosing Maximum Coefficients for Steering Vectors via ALERT

Max Loeffler, Sponsored by Plastic Labs

## Abstract

Language model behavior can be guided by applying steering vectors, which are added directly to the residual stream. The strength of the effect is controlled by a scalar multiplier on the vector, called the steering coefficient. This coefficient has previously been chosen through manual testing, as too high a value will cause the model outputs to become incoherent. We construct a metric over model outputs that detects this incoherence, and use it to automatically choose the maximum coherent steering coefficient. We call this method ALERT, for Automatic Log-probability Evaluation for Representation Tuning. We evaluate ALERT against human judgements across a range of steering vectors, finding it to be a reliable and effective approach to steering coefficient selection. We also extensively explore the input parameters for the method, making a strong recommendation for robust defaults.

## Introduction

Language models can be controlled by directly manipulating their internal representations - a technique known as representation engineering or "rep eng." The key insight is that by modifying the residual stream in a principled way, we can guide the model's behavior without fine-tuning or additional training.

A common approach to rep eng is to create "steering vectors" from the difference between residual stream representations of desired and undesired behaviors. We prompt the model with positive examples (desired behavior) and negative examples (undesired behavior), then save the contents of the residual stream for each. We subtract the negative examples from the positive examples to get pairwise differences. Taking the mean of these differences yields a steering vector that can be added back to the residual stream during generation to guide the model's outputs.

The strength of the steering effect depends on the magnitude of the vector, which is controlled by a scalar multiplier called the steering coefficient. A crucial challenge emerges: how do you choose the right coefficient? Too small, and the vector has little to no effect. Too large, and the model outputs become incoherent. The best approach has been to iteratively test coefficients and evaluate outputs by hand, but this approach is time-consuming and imprecise.

To address this challenge, we present a novel method for automatically determining the maximum steering coefficient, ALERT: Automatic Log-probability Evaluation for Representation Tuning. ALERT iterates across a range of coefficients, generating completions from the same prompt. It then uses the unsteered model to calculate average log probabilities for each completion. We find this metric exhibits a sharp change when outputs become incoherent. By setting the maximum coefficient to this discontinuity, ALERT captures the full range of useful coefficients and minimizes the inclusion of incoherent coefficients. We evaluate this method against human judgements across a range of steering vectors, finding it to be a reliable and effective approach to steering coefficient selection. We also extensively explore the input parameters for the method, making a strong recommendation for robust defaults.

## Method

### Technical Setup

We use SAEs in the generation of high-quality steering vectors. We wish to find a method that works on frontier LLMs, so we use the largest LLM with publicly available SAEs. At the time of our work, this is Llama 3 8B base with EleutherAI SAEs. These SAEs attempt to reconstruct the residual stream after each layer, and have inner dimension 32x that of the model's residual stream. There is one SAE per layer. For certain experiments, we also use the instruction-tuned version of LLaMA 3 8B, as the SAEs transfer fairly well.

We generate steering vectors via Contrastive Activation Addition (CAA) in the SAE latent space. We process positive and negative activation pairs by finding the midpoint between them, then subtracting the midpoint from both activations. Then, we aggregate the pairs by using PCA to extract the first principal component.

### Manually Finding the Maximum Coefficient

In order to develop our automatic method, we first consider the manual process of evaluating steering vectors. Here, we slowly increase the steering coefficient, and observe the effect on the model's outputs. There are two key inflection points: when the model begins to exhibit steered behavior, and when it becomes incoherent. These two points can serve as the minimum and maximum steering coefficients, as they bound the range of coherent steered outputs. We notice that when hand selecting coefficients, the distance between the minimum and maximum is much larger than the distance between the minimum and zero. In other words, finding a minimum coefficient eliminates a fairly small portion of the coefficient range. In contrast, finding a maximum coefficient eliminates all values between it and infinity! Therefore, we are mostly interested in finding the maximum coefficient.

### Metric Selection

In order to automate the above manual process, we seek to replace the human evaluation of output with a metric. This will allow us to automatically iterate over the coefficient range, searching for the two inflection points. Intuitively, we expect this metric to be related to the quality of the model's predictions, as steering shifts those predictions considerably. We consider metrics used to evaluate the quality of a model during and after training. We also notice that the model's over-steered incoherent outputs are nearly always repetitive. Therefore, we also consider metrics that reflect repetition. Initially, we test the following metrics:

- Cross entropy loss
- Perplexity
- Sequence entropy
- A token frequency-based repetition score

We find that entropy and repetition score clearly indicate the point of incoherence, tracking each other closely. However, they do not track the minimum threshold for steered behavior. In contrast, perplexity and cross entropy loss increase when the model starts to exhibit steered behavior. However, they do not clearly indicate the maximum coefficient, remaining fairly flat until well past the point of incoherence. This is somewhat surprising! We hypothesize that the same factors which make the model incoherent also make it very confident, particularly when it is repeating the same tokens. As these metrics are a measure of model confidence, this would keep them low. We still wish to find a metric that can be used to detect the both coefficients, so we continue to evaluate metrics.

As we think the steered model is overly confident, we try using an unsteered version of the model to compute these metrics instead. We also try combining metrics, as different metrics effectively capture different inflection points. Cross entropy loss is a bit spiky, so we exclude it from the combination. We combine perplexity with entropy and repetition score. Using the unsteered model as an evaluator, we test:

- Cross entropy loss
- Perplexity
- Perplexity with entropy
- Perplexity with repetition bias

The combined metrics require a coefficient to determine how important each term is in the final score. We find this parameter to be sensitive and context dependent, so we discard these combinations. Cross entropy loss and perplexity maintain their previous behavior, signaling the minimum coefficient, and now also exhibit clear spikes at the point of incoherence - this is very promising! However, they both drop back down after this inflection point, as the model's repetitions eventually induce high confidence even in the unsteered evaluator. This complicates automatic extraction of the inflection point, as each metric is noisy, and has many spikes; if we use one of these metrics, we might occasionally choose the wrong spike. We want a metric that exhibits a clear phase shift, like entropy or repetition score did.

We consider that the exponentiation during the calculations of perplexity and cross entropy loss limits the effect of outliers on the overall score, but we want a metric that is highly sensitive to outliers. Without exponentiation, perplexity describes how likely a sequence is, so we try calculating the probability of a sequence directly. However, any sequence probability is vanishingly small, much smaller than floating point precision. However, the probability of each token is calculated by exponentiating the logits, or logprobs. Therefore, instead of calculating sequence probability by exponentiating logprobs into probabilities and multiplying across the sequence, we instead use the logprobs themselves. We aggregate over the sequence by taking the mean over each token's logprobs.

This metric works! It exhibits two clear phase shifts: a slight rise and plateau when the steering vector takes effect, then a sharp spike at the point of incoherence. We can now use this metric to automatically find the maximum steering coefficient. Below, see a comparison of all tested metrics with the unsteered model as evaluator. All metrics have been normalized to the same scale.

![Comparison of Metrics for Finding Maximum Activation Value for Steering Vector](metrics-comparison.png)

Below, all tested metrics in the original configuration, with steered model as evaluator.

![Comparison of Metrics for Finding Maximum Activation Value for Steering Vector, Steered Evaluation](metrics-comparison-steered.png)

With a metric in hand, we can now fully define our automatic method.

### Automatic Log-probability Evaluation for Representation Tuning

We call our method ALERT, or Automatic Log-probability Evaluation for Representation Tuning. As with the manual process, ALERT begins with iterative generation of sequences at increasing steering coefficients. For each sequence, we compute the average log probability using the unsteered model. This metric, as shown above, clearly shows inflection points for both steering and incoherence. We extract the incoherence point by defining a simple threshold: return the first point where the metric increases at least 20% above baseline. This 20% threshold is somewhat arbitrary - the metric typically spikes dramatically enough that the exact percentage isn't crucial. We choose 20% because we occasionally see earlier, smaller spikes above baseline, and this threshold reliably clears them. Higher thresholds (e.g., 30% or 40%) also work well, though they include slightly more incoherent coefficients. We do not implement a method for extracting the minimum coefficient from the metric, and leave this to future work.

### Optimal Generation Parameters

Through extensive testing, we identify the following configuration optimal for generating steered sequences in ALERT:

- Temperature: 0 (greedy decoding)
- Repetition penalty: 1.1
- Token count: 32
- Prompt: Shared prefix between positive and negative steering prompts

See Appendix A for detailed parameter analysis and justification.

Note that these parameters are specifically optimized for clearly indicating the inflection points - they are not optimal for steering. Additionally, these optimal values are specific to our technical setup with LLaMA 3 and EleutherAI SAEs; while we expect them to transfer, other models or steering approaches might benefit from different parameters.

## Results and Evaluation

### Ground Truth Validation

We validate ALERT against human judgements across multiple diverse prompts. We select prompts to represent the following categories:

- Object Descriptive Steering: "I am talking about the Golden Gate Bridge in San Francisco, California"
- Abstract Descriptive Steering: "I am talking about wedding ceremonies and traditions"
- Character Trait Steering: "I am speaking with empathy and emotional understanding"
- Task Steering: "I am writing Python code with detailed comments"
- Identity Steering: "I am a conspiracy theorist"

For each prompt, we plot the ALERT metric and the human-annotated incoherence threshold, which is marked with a star.

![Human Coherence Ratings vs Metric Threshold](human-labels.png)

For all prompts, we clearly see that the metric used in ALERT aligns well with human judgements of coherence, showing a sharp transition shortly after the human-annotated threshold.

Below is a sample of the outputs at and around the human-annotated threshold for each prompt.

````text
Prompt: talking about the Golden Gate Bridge in San Francisco, California

Output 3: I am talking about the famous bridge that is located in San Francisco, California. The bridge was built in 1937 and it took five years to build this beautiful bridge. It

Output 4: I am talking about the famous bridge that is located in San Francisco, California. The bridge was built in 1935 and it took a total of 4 years to build this

Output 5: I am talking about the famous bridge that was built in San Francisco, California which is located in the city of this beautiful bridge that was constructed in the city of this bridge that was

---

Prompt: talking about wedding ceremonies and traditions

Output 3: I am talking about the wedding traditions that are so deeply rooted in our culture and have been passed down from generation to generation. The ones that we all love, cherish and still find

Output 4: I am talking about the wedding traditions that are so beautiful and romantic, but also those that have been a part of many weddings in our country. We have seen them all, from

Output 5: I am talking about the wedding traditions that have been so much different to those weddings and how many of them we have seen in our own countries, as well as the ones that are

---

Prompt: speaking with empathy and emotional understanding

Output 3: I am talking about the ability to see and understand your own emotional and spiritual needs, as well as those of others. I have a deep empathy for the human condition and can help

Output 4: I am talking about the ability to understand and empathise with you, recognising the emotional dynamics of your experience as a compassionate presence and empathy in understanding how we can assist you to

Output 5: I am talking about empathing and understanding the emotional qualities of compassion and empathy in your capacity to assist you as a empathic empath and empathing deeply and supporting you as a compassionate

---

Prompt: writing Python code with detailed comments

Output 3: I am talking about the following code snippet:

```
def get_code(self):     return self.code def get_docstring(self):     """... ``` I have a question: Is

Output 4: I am talking about the following code for a colleague who needs to use it in Python (pydoc). The code is commented using the docstring style. I would like to review

Output 5: I am talking about the following code to generate a good practice for generating the following comments below:


 Best Python developers using the following best comments to generate the code for the following

---

Prompt: a conspiracy theorist

Output 4: I am talking about the term not in the sense that I believe that there are “secret” conspiratorial elements to the word, but rather in the sense that I accept the fact

Output 5: I am talking about the term not in the sense that I believe many of my colleagues admit to the label, although I acknowledge the reality of the term not in the sense that I

Output 6: I am talking about the term not before I admit the label, or at least believe that I accept the term before I acknowledge the term that I believe that I suspect that I believe
````

### Comparison with Single-Token Methods

Recent work by [Chalnev, Siu, Conmy](https://arxiv.org/abs/2411.02193) has proposed using a cross-entropy loss threshold on a single-token generation to determine maximum coefficient values. While computationally efficient, our analysis shows this approach sacrifices precision for speed.

![Single Token Cross Entropy Loss (theirs) vs. 32 Token Avg Logprobs (ours)](single-token-comparison.png)

The graph above shows multiple trials, each with a different prompt, using their single-token method. The metric they measure is charted, with each maximum coefficient they select marked with a red circle. Their method does not show clear structure in the metric, nor consistency in chosen inflection points across prompts. We highlight, in purple, a selected prompt, and show our method's behavior on the same prompt. ALERT closely matches the ground truth on this prompt, while their method yields a much lower maximum coefficient.

ALERT is more computationally expensive, but we think the improved reliability justifies this expense for many applications. While single-token methods may be suitable for rapid prototyping or resource-constrained environments, our multi-token approach provides the precision needed for reliable maximum coefficient selection.

### Transferability to Instruction-tuned Models

SAEs and steering vectors made for base models transfer well to their instruction-tuned counterparts. We investigate if ALERT can be used with base model SAEs and steering vectors when applied to instruction-tuned models. Below, we show the ALERT metric's behavior across the two types of models. We examine all possible combinations of generator and evaluator models across base and instruction-tuned models, as well as between steered evaluation and unsteered evaluators.

![Effect of Model Type on Steering Measurement](model-type-comparison.png)

ALERT transfers well. The original configuration of base model for generator and evaluator, with steered evaluator, provides the clearest signal. However, the differences are minimal, and ALERT works reliably for all variants.

## Limitations and Future Work

The method of generating steering vectors we describe in "Technical Setup" was state of the art at the time of our work, but is now far from optimal. We do not test other methods of generating steering vectors, and cannot confirm that ALERT functions as well with different steering vector generation methods.

We do not implement a method for extracting the minimum coefficient from the ALERT metric, and leave this to future work.

We do not test with other models or SAEs, though we expect the method to work similarly.

## Conclusion

Finding optimal steering coefficients doesn't have to be a manual process. With careful attention to generation parameters and the right metric, we can automatically detect when a steering vector transitions from effective to incoherent. Our method, ALERT, enables more systematic and scalable approaches to model steering, including the ability to build systems on top of automatically generated steering vectors. We are excited to see what future work can do with this method.

## Appendix A: Detailed Generation Parameter Analysis

Token generation parameters significantly impact ALERT metric reliability. We investigate the following parameters in detail: Temperature, Repetition Penalty, Tokens Generated, and Prompt.

### Temperature

Our investigation of temperature revealed two key problems with higher temperature values:

1. Less Sharp Transitions: Higher temperatures make the metric curve smoother, making it harder to detect the precise point where steering becomes incoherent. See graph below:

![Effect of Temperature on Steering Measurement](temperature-comparison.png)

2. Increased Variability: Even at relatively low temperatures, multiple trials show significant noise. The following graph demonstrates this variability with multiple runs at temperature 0.5:

![Multiple Trials at Temperature 0.5](temperature-trials.png)

These issues led us to use greedy decoding (temperature = 0), which provides the cleanest and most reliable signal.

### Repetition Penalty

The repetition penalty value needs careful tuning to maintain metric clarity:

![Effect of Repetition Penalty on Steering Measurement](repetition-penalty-comparison.png)

We found that a repetition penalty of 1.1 provides optimal results by balancing two issues:

- No penalty (1.0) causes the initial "plateau" to become spiky and uneven, creating artificial metric jumps
- Higher penalties smooth out the curve too much, obscuring the crucial steering spike where the model becomes incoherent. Oddly, some higher penalties also cause the plateau to become spiky.

### Tokens Generated

The number of tokens generated significantly impacts metric reliability:
![Effect of Token Count on Coefficient vs Avg Logprobs](token-count-comparison.png)

Key findings:

- Curves begin to sharpen around 12 tokens
- Optimal results appear at 20+ tokens
- Lower token counts produce flatter, less useful curves
- Higher token counts sharpen further, but offer diminishing returns relative to computational cost

### Prompt

We tested various prompt approaches:

1. Shared prefix (recommended)
2. Full positive prompt
3. Full negative prompt
4. Directly referencing prompt subject without using the positive prompt
5. Generic prefix
6. Unrelated concept

![Comparison of Test Prompts for Finding Maximum Activation Value for Steering Vector, Steered Evaluation](test-prompt-trials.png)

Results were inconclusive. The different prompt types did not yield significantly different results. We select shared prefix, as intuitively, it seems like it should lead to the most predictable steering effects.
