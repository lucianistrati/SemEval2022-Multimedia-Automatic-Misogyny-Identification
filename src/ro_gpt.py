from transformers import AutoTokenizer, AutoModelForCausalLM
# for finetuning a gpt2 for text generation (or use it as raw) with the following 3 use cases

# social media posts generation
# answers on facebook chatbots generation
# chatbotting

# either GPT2 or another Question answering or text generation model
# https://colab.research.google.com/github/philschmid/fine-tune-GPT-2/blob/master/Fine_tune_a_non_English_GPT_2_Model_with_Huggingface.ipynb
# https://towardsdatascience.com/fine-tuning-gpt2-for-text-generation-using-pytorch-2ee61a4f1ba7
# https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a

# text generation pe engleza(gpt2): gpt2
# question answering squad2 pe engleza (roberta): deepset/roberta-base-squad2

# perceiver multimodal recommended by dave to be tried

# ro_gpt2_models = ["readerbench/RoGPT2-base", "readerbench/RoGPT2-large", "readerbench/RoGPT2-medium"]
# tokenizer = AutoTokenizer.from_pretrained('readerbench/RoGPT2-base')
# model = AutoModelForCausalLM.from_pretrained('readerbench/RoGPT2-base')
# inputs = tokenizer.encode("Este o zi de vara", return_tensors='pt')
# text = model.generate(inputs, max_length=1024,  no_repeat_ngram_size=2)
# print(tokenizer.decode(text[0]))

"""
input:
Este o zi de vara

output:
Este o zi de vara, cu soare, în care soarele strălucește în toată splendoarea sa. 
Este ziua în amiaza mare, când soarele răsare la amiază. În această zi, soarele este la fel de strălucitor ca și luna.<|endoftext|>
"""

