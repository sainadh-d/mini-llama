# Mini Llama

This is my attempt to understand Meta's Llama Model Architecture and also train a mini Llama.

## Resources
1. https://github.com/karpathy/build-nanogpt for the code to fetch data and also to train.
2. https://github.com/meta-llama/llama3 for the Llama Model Architecture
3. https://arxiv.org/abs/2302.13971 Original Llama Paper
4. https://github.com/karpathy/llama2.c for Reference Llama Implementation and Tokenizer

## Process

1. Get the Llama Model from [Llama Repo](https://github.com/meta-llama/llama3)
2. Get the data. (Chosen Fineweb-Edu the same one used in build-nanogpt)
3. Write the code to and train. (The code to get the data and to train was mostly a fork of [build-nanogpt](https://github.com/karpathy/build-nanogpt) by Andrej Karpathy and made to fit the Llama Model)
4. Generate from the Model
5. Evaluate (HellaSwag)

## Model

Model parameters used are similar to GPT2 except for Vocab Size which is 32k for Llama Tokenizer.

This resulted in a model of 109M parameters.

```python
@dataclass
class MiniLlamaArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    dropout: float = 0.0
```

## Results
Results can be seen in [infer.ipynb](infer.ipynb) and [visualize.ipynb](visualize.ipynb)

Even though the model is slightly smaller than GPT2 (124M vs 109M) it performed 
- almost same on HellaSwag Eval 
- better than GPT2 on FineWeb Edu

On FineWeb Edu
- Min Train Loss: 2.555338 (2.922 for GPT2 from Karpathy's video)
- Min Validation Loss: 2.7095 (3.0726 for GPT2)

HellaSwag Accurarcy
- 30.34% (30.68% for GPT2)

# Sample Generation
```
generate(model, "To stay healthy, I have to")

Sample 0: To stay healthy, I have to eat a whole foods diet – like chicken, eggs, grains, fruits and nuts. This includes all of the “healthy” foods, soy, dairy products, and fruits and vegetables, as well as refined grains, meats, milk, and dairy products. These foods are good. You can keep them part of a healthy diet by eating a whole food

Sample 1: To stay healthy, I have to work all day. But I also require a little push back from people who have been hurt.
How many times have you tried to help an elder baby?
I have done so many times in my life.
How many times have you thought you were cared for?
An older person’s decision to stay home from work can also result in a high-risk situation called drowning. This is a serious medical emergency.

Sample 2: To stay healthy, I have to be healthy!
In this blog post, I will look at a few key health benefits of eating healthily.
1. Eating Healthy
Healthy foods are great for your whole body and helps fight off disease. One easy way to get healthy is by exercising and eating fruits and vegetables. Healthy fruits and vegetables are great for your brain because they can help regulate your nerv

Sample 3: To stay healthy, I have to do things that help me to stay active, sleep better, and take care of myself. For me, I’ll do things that help me a lot, and for most of us, those help me to look at things differently. I’ll do things that help me to better myself, and to get more exercise.
It really is a really important part of my life because it’s the building block of growth.
Human beings
```

