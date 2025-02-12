import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset


def main() -> None:

    # Load the model.
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip", use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
    model.cuda()

    # Load an example input image.
    dataset = load_dataset("hf-internal-testing/example-documents", split="test")
    image = dataset[1]["image"]
    inp = processor(image, return_tensors="pt").pixel_values.cuda()

    # Create an example input prompt.
    task_prompt = "<s_rvlcdip>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

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

        # Push current scaled input through the model.
        outputs = model(scaled_inputs[i], decoder_input_ids=decoder_input_ids)

        # Compute the gradient with respect to the input. outputs[i, j, k] is
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
    print(integrated_gradients.shape)


if __name__ == "__main__":
    main()
