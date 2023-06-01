import torch

"""
input/instruction/outputからpromptを作る
"""
def generate_prompt(instruction, input=None, response=None):
  def add_escape(text):
    return text.replace('### Response', '###  Response')

  if input:
    prompt = f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{add_escape(instruction.strip())}

### Input:
{add_escape(input.strip())}
""".strip()
  else:
    prompt = f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{add_escape(instruction.strip())}
""".strip()

  if response:
    prompt += f"\n\n### Response:\n{add_escape(response.strip())}<|endoftext|>"
  else:
    prompt += f"\n\n### Response:\n"

  return prompt

def get_response(text):
  marker = f"### Response:\n"
  pos = text.find(marker)
  if pos == -1:  # marker not found
      return None
  return text[pos + len(marker):].strip()


def qa2(tokenizer, model, instruction, context=None):
  prompt = generate_prompt(instruction, context)

  batch = tokenizer(prompt, return_tensors='pt')
  with torch.cuda.amp.autocast():
    output_tokens = model.generate(
        **batch,
        max_new_tokens=256,
        temperature = 0.7,
        repetition_penalty=1.05
    )

  text = tokenizer.decode(output_tokens[0],pad_token_id=tokenizer.pad_token_id, skip_special_tokens=True)
  return get_response(text)
