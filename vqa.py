import re
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset


def get_integrated_gradients(model, inp, decoder_input_ids):
    # Define inputs to integrated gradients. Variables have the same names as
    # the official repository. Assume the baseline is zero.
    target_index_label = 0
    steps = 5 

    # Compute the scaled inputs. Assume the baseline is zero.
    scaled_inputs = [(i / steps) * inp for i in range(steps + 1)]

    # Require gradients for each input.
    for scaled_inp in scaled_inputs:
        scaled_inp.requires_grad = True
        scaled_inp.retain_grad()

    # Compute the gradient with respect to each scaled input.
    grads = []
    for i in range(steps + 1):
        model.zero_grad()
        scaled_inputs[i].grad = None

        # Push current scaled input through the model.
        outputs = model(scaled_inputs[i], decoder_input_ids=decoder_input_ids)

        # Compute the gradient with respect to the input. logits[i, j, k] is
        # the probability of generating the k-th word, in the j-th output 
        # position of the i-th batch. Since there is only one batch, i=0.
        # To simplify things, also only consider the first output word.
        outputs.logits[0, 0, target_index_label].backward()

        # Save the gradient.
        grads.append(scaled_inputs[i].grad.detach())

    # Convert the gradients to a tensor.
    grads = torch.concat(grads, dim=0)

    # Compute the integrated gradients.
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = torch.mean(grads, axis=0)
    integrated_gradients = inp * avg_grads

    return integrated_gradients


def main() -> None:

    # Load the model.
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model.cuda()

    # Load an example input image.
    dataset = load_dataset("hf-internal-testing/example-documents", split="test")
    image = dataset[0]["image"]
    inp = processor(image, return_tensors="pt").pixel_values.cuda()

    # Create an example input prompt.
    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    question = "When is the coffee break?"
    prompt = task_prompt.replace("{user_input}", question)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

    # Generate the model output.
    outputs = model.generate(
        inp,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    output_sequence_ids = outputs.sequences[0]
    print("output: '{processor.decode(output_sequence_ids)}'")

    # For each output token, get the integrated gradients with respect to the input image.
    for i in range(len(decoder_input_ids), len(output_sequence_ids) + 1):
        input_ids = output_sequence_ids[:i]
        integrated_gradients = get_integrated_gradients(model, inp, input_ids.unsqueeze(0))
        print(f"integrated gradient for '{processor.decode(input_ids)}':", integrated_gradients.shape)



if __name__ == "__main__":
    main()
