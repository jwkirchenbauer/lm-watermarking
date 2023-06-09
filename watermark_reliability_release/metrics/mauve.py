"""
MAUVE 
Adapted from the eval script in https://github.com/XiangLi1999/ContrastiveDecoding
requirement: pip install mauve-text (https://github.com/krishnap25/mauve)
"""

from transformers import AutoTokenizer
import mauve


def get_mauve_score(
    p_text, q_text, max_len=128, verbose=False, device_id=0, featurize_model_name="gpt2"
):
    """
    p_text: reference completion
    q_text: output completion
    """
    print(f"initial p_text: {len(p_text)}, q_text: {len(q_text)}")

    ## preprocess: truncating the texts to the same length
    tokenizer = AutoTokenizer.from_pretrained(featurize_model_name)
    # tokenize by GPT2 first.
    x = tokenizer(p_text, truncation=True, max_length=max_len)["input_ids"]
    y = tokenizer(q_text, truncation=True, max_length=max_len)["input_ids"]

    # xxyy = [(xx, yy) for (xx, yy) in zip(x, y) if len(xx) == max_len and len(yy) == max_len]
    # NOTE check with Manli, is this ok?
    xxyy = [
        (xx, yy)
        for (xx, yy) in zip(x, y)
        if (len(xx) <= max_len and len(xx) > 0) and (len(yy) <= max_len and len(yy) > 0)
    ]
    x, y = zip(*xxyy)

    # map back to texts.
    p_text = tokenizer.batch_decode(x)  # [:target_num]
    q_text = tokenizer.batch_decode(y)  # [:target_num]
    print(f"remaining p_text: {len(p_text)}, q_text: {len(q_text)}")

    # call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to 256 tokens
    out = mauve.compute_mauve(
        p_text=p_text,
        q_text=q_text,
        device_id=device_id,
        max_text_length=max_len,
        verbose=verbose,
        featurize_model_name=featurize_model_name,
    )
    # print(out)

    return out.mauve
