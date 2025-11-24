"""
Parser for extracting thinking and response segments from DeepSeek-R1 outputs.

DeepSeek-R1 uses <think> tags to mark internal reasoning traces.
This module extracts these segments for analysis.

Note: The research plan mentions <think> tags conceptually, but DeepSeek-R1
actually outputs <think> tags in practice.
"""

import re
from typing import Dict, Optional


def parse_reasoning_output(text: str) -> Dict[str, str]:
    """
    Parse DeepSeek-R1 output to extract thinking and response segments.
    
    Args:
        text: Raw model output string
        
    Returns:
        Dictionary with keys:
            - 'thinking_segment': Text inside <think> tags
            - 'response_segment': Text after </think> tag
            - 'full_text': Original text
            - 'has_reasoning': Boolean indicating if reasoning tags were found
    """
    result = {
        "thinking_segment": "",
        "response_segment": "",
        "full_text": text,
        "has_reasoning": False,
    }
    
    # Pattern to match <think>...</think> tags
    # DeepSeek-R1 uses <think> tags (the model's actual output format)
    pattern = r'<think>(.*?)</think>'
    
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        result["thinking_segment"] = match.group(1).strip()
        result["has_reasoning"] = True
        
        # Extract response segment (everything after the closing tag)
        end_pos = match.end()
        result["response_segment"] = text[end_pos:].strip()
    else:
        # No reasoning tags found - treat entire text as response
        result["response_segment"] = text.strip()
        result["has_reasoning"] = False
    
    return result


def get_token_indices_for_segments(
    text: str,
    tokenizer,
    thinking_segment: str,
    response_segment: str,
) -> Dict[str, tuple[int, int]]:
    """
    Map text segments to token indices for analysis.
    
    Args:
        text: Full text
        tokenizer: Tokenizer instance
        thinking_segment: Thinking segment text
        response_segment: Response segment text
        
    Returns:
        Dictionary with 'thinking' and 'response' keys, each containing
        (start_idx, end_idx) tuple of token indices
    """
    # Tokenize full text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_text = tokenizer.decode(tokens, skip_special_tokens=False)
    
    # Find positions in original text
    # DeepSeek-R1 uses <think> tags
    thinking_start = text.find("<think>")
    thinking_end = text.find("</think>")
    
    if thinking_end != -1:
        response_start = thinking_end + len("</think>")
    else:
        response_start = 0
    
    # Tokenize segments separately to get their lengths
    if thinking_segment:
        thinking_tokens = tokenizer.encode(thinking_segment, add_special_tokens=False)
        thinking_token_len = len(thinking_tokens)
    else:
        thinking_token_len = 0
    
    if response_segment:
        response_tokens = tokenizer.encode(response_segment, add_special_tokens=False)
        response_token_len = len(response_tokens)
    else:
        response_token_len = 0
    
    # Find token indices by searching for segment boundaries
    # This is approximate - we'll use a more robust method
    full_tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    token_ids = full_tokens["input_ids"][0].tolist()
    
    # Decode to find where segments start
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    
    # Find the position of the opening tag in tokenized space
    # DeepSeek-R1 uses <think> tags
    tag_start = "<think>"
    tag_end = "</think>"
    
    # Simple approach: tokenize with tags and find boundaries
    # Tokenize the prefix up to thinking
    if tag_start in text:
        prefix = text.split(tag_start)[0]
        prefix_tokens = tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        thinking_start_idx = len(prefix_tokens)
        
        # Tokenize thinking segment
        if thinking_segment:
            thinking_tokens = tokenizer(thinking_segment, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            thinking_end_idx = thinking_start_idx + len(thinking_tokens)
        else:
            thinking_end_idx = thinking_start_idx
        
        # Response starts after closing tag
        response_start_idx = thinking_end_idx + 1  # +1 for closing tag tokens
        response_end_idx = response_start_idx + response_token_len if response_token_len > 0 else len(token_ids)
    else:
        # No reasoning tags - entire output is response
        thinking_start_idx = 0
        thinking_end_idx = 0
        response_start_idx = 0
        response_end_idx = len(token_ids)
    
    return {
        "thinking": (thinking_start_idx, thinking_end_idx),
        "response": (response_start_idx, response_end_idx),
    }

