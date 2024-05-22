# """
# Instruction-tuning with LoRA on the Alpaca dataset.

# Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
# `torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
# """
# import sys
# from pathlib import Path
# import os
# import time

# import lightning as L
# import numpy as np
# import torch

# # support running without installing as a package
# wd = Path(__file__).parent.parent.resolve()
# sys.path.append(str(wd))

# from generate import generate
# from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
# from lit_llama.model import LLaMA, LLaMAConfig
# from lit_llama.tokenizer import Tokenizer
# from scripts.prepare_alpaca import generate_prompt


# instruction_tuning = True
# eval_interval = 100
# save_interval = 100
# eval_iters = 100
# log_interval = 10

# # Hyperparameters
# learning_rate = 3e-4
# batch_size = 128
# micro_batch_size = 4
# gradient_accumulation_iters = batch_size // micro_batch_size
# assert gradient_accumulation_iters > 0
# max_iters = 25000 * 3 // micro_batch_size
# weight_decay = 0.0
# max_seq_length = 1024  # see scripts/prepare_alpaca.py
# lora_r = 8
# lora_alpha = 16
# lora_dropout = 0.05
# warmup_iters = 100


# def main(
#     data_dir: str = "data/highway-easy", 
#     pretrained_path: str = "checkpoints/lit-llama/13B/lit-llama.pth",
#     tokenizer_path: str = "checkpoints/lit-llama/tokenizer.model",
#     out_dir: str = "out/lora/alpaca",
# ):

#     fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-true")
#     fabric.launch()
#     fabric.seed_everything(1337 + fabric.global_rank)

#     if fabric.global_rank == 0:
#         os.makedirs(out_dir, exist_ok=True)

#     train_data, val_data = load_datasets(data_dir=data_dir)

#     config = LLaMAConfig.from_name("13B")
#     config.block_size = max_seq_length

#     checkpoint = torch.load(pretrained_path)

#     with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
#         model = LLaMA(config)
#         # strict=False because missing keys due to LoRA weights not contained in checkpoint state
#         model.load_state_dict(checkpoint, strict=False)
    
#     mark_only_lora_as_trainable(model)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     model, optimizer = fabric.setup(model, optimizer)
#     train(fabric, model, optimizer, train_data, val_data, tokenizer_path, out_dir)

#     # Save the final LoRA checkpoint at the end of training
#     checkpoint = lora_state_dict(model)
#     fabric.save(os.path.join(out_dir, "highway-easy-r16-finetuned.pth"), checkpoint)


# def train(
#     fabric: L.Fabric,
#     model: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     train_data: np.ndarray,
#     val_data: np.ndarray,
#     tokenizer_path: str,
#     out_dir: str,
# ) -> None:
#     """The training loop.

#     Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
#     """
#     step_count = 0
    
#     for iter_num in range(max_iters):

#         if step_count <= warmup_iters:
#             # linear warmup
#             lr = learning_rate * step_count / warmup_iters
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr

#         t0 = time.time()

#         input_ids, targets = get_batch(fabric, train_data)
#         with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
#             logits = model(input_ids[0].unsqueeze(0))
#             loss = loss_fn(logits, targets)
#             fabric.backward(loss / gradient_accumulation_iters)
#             tokenizer = Tokenizer(tokenizer_path)
#             generated_token_ids = logits.argmax(dim=-1)
#             generated_text = tokenizer.decode(generated_token_ids.squeeze())
#             fabric.print("Generated response:", generated_text)

#         if (iter_num + 1) % gradient_accumulation_iters == 0:
#             optimizer.step()
#             optimizer.zero_grad()
#             step_count += 1
                
#             if step_count % eval_interval == 0:
#                 val_loss = validate(fabric, model, val_data, tokenizer_path)
#                 fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
#                 fabric.barrier()

#             if step_count % save_interval == 0:
#                 print(f"Saving LoRA weights to {out_dir}")
#                 # We are only saving the LoRA weights
#                 # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
#                 checkpoint = lora_state_dict(model)
#                 fabric.save(os.path.join(out_dir, f"iter-r16-{iter_num:06d}-ckpt.pth"), checkpoint)

#         dt = time.time() - t0
#         if iter_num % log_interval == 0:
#             fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


# def generate_response(model, instruction, tokenizer_path):
#     tokenizer = Tokenizer(tokenizer_path)
#     sample = {"instruction": instruction, "input": ""}
#     prompt = instruction
#     if instruction_tuning:
#         prompt = generate_prompt(sample)
#     encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

#     output = generate(
#         model,
#         idx=encoded,
#         max_seq_length=max_seq_length,
#         max_new_tokens=100,
#     )
#     output = tokenizer.decode(output)
#     return output # output.split("### Response:")[1].strip()


# @torch.no_grad()
# def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, tokenizer_path: str) -> torch.Tensor:
#     fabric.print("Validating ...")
#     model.eval()
#     losses = torch.zeros(eval_iters)
#     for k in range(eval_iters):
#         input_ids, targets = get_batch(fabric, val_data)
#         logits = model(input_ids)
#         loss = loss_fn(logits, targets)
#         losses[k] = loss.item()
#     out = losses.mean()

#     # # produce an example:
#     # instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
#     # output = generate_response(model, instruction, tokenizer_path)
#     # fabric.print(instruction)
#     # fabric.print(output)

#     model.train()
#     return out.item()

# def loss_fn(logits, targets):
#     # shift the targets such that output n predicts token n+1
#     logits = logits[..., :-1, :].contiguous()
#     targets = targets[..., 1:].contiguous()
#     loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
#     return loss
    

# def get_batch(fabric: L.Fabric, data: list):
#     ix = torch.randint(len(data), (micro_batch_size,))

#     input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
#     labels = [data[i]["labels"].type(torch.int64) for i in ix]

#     max_len = max(len(s) for s in input_ids)

#     def pad_right(x, pad_id):
#         # pad right based on the longest sequence
#         n = max_len - len(x)
#         return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

#     x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
#     y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
#     x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
#     return x, y


# def load_datasets(data_dir):
#     train_data = torch.load(os.path.join(data_dir, "highway-easy-train.pt"))
#     val_data = torch.load(os.path.join(data_dir, "highway-easy-test.pt"))
#     return train_data, val_data


# if __name__ == "__main__":
#     # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
#     # torch.backends.cuda.enable_flash_sdp(False)
#     torch.set_float32_matmul_precision("high")
    
#     from jsonargparse.cli import CLI

#     CLI(main)
"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import sys
from pathlib import Path
import os
import time
from torch.utils.tensorboard import SummaryWriter

import lightning as L
import numpy as np
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from lit_llama.utils import lazy_load, llama_model_lookup
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

instruction_tuning = True
eval_interval = 100
save_interval = 100
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 3e-4
batch_size = 128
micro_batch_size = 4
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_iters = 40000 * 17 // micro_batch_size
weight_decay = 0.0
max_seq_length = 1024  # see scripts/prepare_alpaca.py
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_iters = 100


def main(
    data_dir: str = "data/highway-complicate-50000", 
    pretrained_path: str = "checkpoints/lit-llama/13B/lit-llama.pth",
    lora_path:Path = Path("out/50000-complicate/highway-complicate-relative-50000-finetuned.pth"),
    tokenizer_path: str = "checkpoints/lit-llama/tokenizer.model",
    out_dir: str = "out/50000-complicate",
):

    fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-true")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

    config = LLaMAConfig.from_name("13B")
    config.block_size = max_seq_length

    checkpoint = torch.load(pretrained_path)

    with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
       model = LLaMA(config)
       # strict=False because missing keys due to LoRA weights not contained in checkpoint state
       model.load_state_dict(checkpoint, strict=False)
    # with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:

    #     name = llama_model_lookup(pretrained_checkpoint)



    #     with fabric.init_module(empty_init=True), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):

    #         model = LLaMA.from_name(name)

    #         model.load_state_dict(pretrained_checkpoint, strict=False)

    #         model.load_state_dict(lora_checkpoint, strict=False)
    
    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    writer = SummaryWriter(log_dir="runs/50000-complicate")
    train(fabric, model, optimizer, train_data, val_data, tokenizer_path, out_dir, writer)

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(out_dir, "highway-complicate-relative-50000-finetuned.pth"), checkpoint)
    writer.close()

def cosine_decay_with_warmup(step_count,
                             learning_rate,
                             max_iters,
                             warmup_learning_rate=0.0,
                             warmup_iters=warmup_iters,
                             hold_base_rate_steps=40000*5/4):
    if(step_count <= warmup_iters):
        #线性增长的实现
        slope = (learning_rate - warmup_learning_rate) / warmup_iters
        warmup_rate = slope * step_count + warmup_learning_rate
        return warmup_rate
    learning_rate_now = 0.5 * learning_rate * (1 + np.cos(np.pi * (step_count - warmup_iters - hold_base_rate_steps) / float(max_iters - warmup_iters - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate_now = np.where(step_count > warmup_iters + hold_base_rate_steps,
                                 learning_rate_now, learning_rate)
    return np.where(step_count > max_iters, 0.0, learning_rate_now)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    tokenizer_path: str,
    out_dir: str,
    writer: SummaryWriter
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    tokenizer = Tokenizer(tokenizer_path)
    for iter_num in range(max_iters):
        lr = cosine_decay_with_warmup(step_count, learning_rate, max_iters)
        # if step_count <= warmup_iters:
        #     # linear warmup
        #     lr = learning_rate * step_count / warmup_iters
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets, oral_input_ids, oral_labels = get_batch(fabric, train_data)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
            logits = model(input_ids)
            # decoded_string = tokenizer.decode(logits.tolist())
            # decoded_string = decoded_string.replace(tokenizer.bos_token, "").replace(tokenizer.eos_token, "")
            # decoded_string = tokenizer.decode(logits[-2], skip_special_tokens=True)
            # print("decoded:",decoded_string)
            # generated_token_ids = logits.argmax(dim=-1)
            # generated_text = tokenizer.decode(generated_token_ids.squeeze())
            # logits_2 = tokenizer.encode(str(generated_text[-1:]), bos=True, eos=True, max_length=1024)
            # print("generated_text:", generated_text)
            # print("\n\ngenerated[-1]:",generated_text[-1:])
            # if(iter_num <= 10):
            #     loss = loss_fn(logits, targets)
            # else:
            #     loss = loss_fn_change(logits, targets, oral_labels)
            # last_period_index = (input_ids == tokenizer.token_to_id(".").unsqueeze(0).unsqueeze(0)).nonzero(as_tuple=True)[1].max()
            # last_logits = logits[:, last_period_index, :]  # 仅选择句号之前的最后一个词的logits
            # decoded_string = tokenizer.decode(last_logits.argmax(dim=-1).tolist())  # 获取最后一个词的预测结果并解码
            # decoded_string = decoded_string.replace(tokenizer.bos_token, "").replace(tokenizer.eos_token, "")
            # print("decoded:", decoded_string)
            # loss = loss_fn(last_logits, targets[:, last_period_index])  # 计算损失值时仅使用最后一个词的预测结果和目标值
            loss = loss_fn(logits, targets)
            fabric.backward(loss / gradient_accumulation_iters)




        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            writer.add_scalar('Loss/train', loss.item(), step_count)
                
            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data, tokenizer_path)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()
                writer.add_scalar('Loss/validation', val_loss, step_count)

            if step_count % save_interval == 0:
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model)
                fabric.save(os.path.join(out_dir, f"iter-50000-complicate-{iter_num:06d}-ckpt.pth"), checkpoint)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


def generate_response(model, instruction, tokenizer_path):
    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": instruction, "input": ""}
    prompt = instruction
    if instruction_tuning:
        prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    # output = generate(
    #     model,
    #     idx=encoded,
    #     max_seq_length=max_seq_length,
    #     max_new_tokens=100,
    # )
    # output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, tokenizer_path: str) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets,oral_input_ids, oral_targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    # instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    # output = generate_response(model, instruction, tokenizer_path)
    # fabric.print(instruction)
    # fabric.print(output)

    model.train()
    return out.item()

def loss_fn(logits, targets):
    # tokenizer_path = "checkpoints/lit-llama/tokenizer.model"
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    
    return loss
# def loss_fn(logits, targets):
#     tokenizer_path = "checkpoints/lit-llama/tokenizer.model"
#     print(targets.shape)
#     tokenizer = Tokenizer(tokenizer_path)
#     generated_token_ids = logits.argmax(dim=-1)
#     logits_str = tokenizer.decode(generated_token_ids.squeeze())
#     print("logits_str:",logits_str)
#     # Convert targets to string and find the position of the last number
#     generated_token_ids = targets.argmax(dim=-1)
#     targets_str = tokenizer.decode(generated_token_ids.squeeze())
#     print("targets_str:",targets_str)
#     # targets_str = tokenizer.decode(targets.tolist())
#     last_num_index = max(targets_str.rfind(str(i)) for i in range(10))

#     # If there is no number in targets, return the original loss
#     if last_num_index == -1:
#         print("None")
#         logits = logits[..., :-1, :].contiguous()
#         targets = targets[..., 1:].contiguous()
#         loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
#         return loss

    # # Otherwise, slice logits and targets to align the last number
    # # logits_text =tokenizer.decode(logits.tolist())
    # # targets_text =tokenizer.decode(targets.tolist())
    
    # logits = logits[..., last_num_index:last_num_index+1, :].contiguous()
    # targets = targets[..., last_num_index+1:last_num_index+2].contiguous()
    # loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    # # generated_token_ids = logits.argmax(dim=-1)
    # # generated_text = tokenizer.decode(generated_token_ids.squeeze())
    # logits_token_ids = logits.argmax(dim=-1)
    # logits_str = tokenizer.decode(logits_token_ids.squeeze())
    # print("targets-num:",last_num_index)
    # print("logits-num:",logits_num)
    
    # return loss   


def loss_fn_change(logits, targets, oral_labels):
    p = SentencePieceProcessor(model_file="checkpoints/lit-llama/tokenizer.model")
    
    tokenizer_path = "checkpoints/lit-llama/tokenizer.model"
    tokenizer = Tokenizer(tokenizer_path)

    total_loss = 0
    batch_size = logits.size(0)

    for i in range(batch_size):
        # Convert logits to token ids and decode to string
        print("i:",i)
        targets_len = len(oral_labels[i])
        generated_token_ids = logits[i].argmax(dim=-1)
        logits_str = tokenizer.decode(generated_token_ids.squeeze())
        # print("logtis_str:",logits_str)
        # last_num_index = find_last_digit_position(logits_str)
        # j=0
        # for j in range(len(generated_token_ids)):
        #     if(generated_token_ids[j] != p.pad_id and generated_token_ids[j] != p.eos_id):
        #         original_lengths += 1
        #     # original_lengths = (generated_token_ids != p.pad_id).sum(dim=-1)
        print("targets_len:",targets_len)
        # # generated_token_ids = targets[i].argmax(dim=-1)
        # targets_str = tokenizer.decode(targets[i].tolist())
        # print("targets_str:",targets_str)
        # Find the position of the last number in logits_str
        #last_num_index = max(logits_str.rfind(str(i)) for i in range(100))
        # print("last_num_index:",last_num_index)
        # If there is no number in logits_str, calculate the original loss
        if targets_len == -1:
            logits_i = logits[i, :-1, :].contiguous()
            targets_i = targets[i, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(logits_i.view(-1, logits_i.size(-1)), targets_i.view(-1), ignore_index=-1)
        else:
            # Convert string index to tensor index
            # last_num_index_tensor = len(logits_str) - last_num_index - 1

            # Get the corresponding elements in logits and targets
            # logits_last_num = logits[i, -last_num_index_tensor-1:-last_num_index_tensor, :]
            # targets_last_num = targets[i, -last_num_index_tensor:-last_num_index_tensor+1]
            logits_last_num = logits[i, targets_len-9:targets_len-1, :]
            targets_last_num = targets[i, targets_len-8:targets_len]
            # Calculate the loss of the last number
            loss = torch.nn.functional.cross_entropy(logits_last_num.view(-1, logits_last_num.size(-1)), targets_last_num.view(-1), ignore_index=-1)
            generated_token_ids = logits_last_num.argmax(dim=-1)
            logits_last_num_str = tokenizer.decode(generated_token_ids.squeeze())
            print("logits_last_num:",logits_last_num_str)
            # generated_token_ids = targets_last_num.argmax(dim=-1)
            # targets_last_num = p.decode(targets_last_num)
            # print("targets_last_num:",targets_last_num)
            # generated_token_ids = targets_last_num.argmax(dim=-1)
            # targets_last_num_str = tokenizer.decode(generated_token_ids.squeeze())
            # print("targets_last_num:",targets_last_num_str)
        total_loss += loss

    return total_loss / batch_size

def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    # last_two_tokens_vectors = y[:, -2:]  
    return x, y,input_ids,labels


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "highway-complicate-50000-relative-train.pt"))
    val_data = torch.load(os.path.join(data_dir, "highway-complicate-50000-relative-val.pt"))
    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    
    from jsonargparse.cli import CLI

    CLI(main)