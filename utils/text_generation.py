"""
Utility functions for generating text and creating needle-in-haystack tests.
Used by both Llama and Gemini model tests.
"""
import random
import string

def generate_filler_text(char_count, style="paragraphs"):
    """
    Generates filler text for needle-in-haystack tests.
    
    Args:
        char_count: The approximate number of characters to generate
        style: The style of text - "paragraphs" (structured text with paragraphs)
               or "simple" (repeating simple text)
    
    Returns:
        Generated text string of approximately char_count length
    """
    if style == "simple":
        # Simple repeating text
        base_phrase = "This is filler text to simulate a large context window. "
        repetitions = (char_count // len(base_phrase)) + 1
        text = (base_phrase * repetitions)[:char_count]
        return text
    
    # Default: Generate more varied content with paragraph structures
    paragraphs = []
    current_length = 0
    
    while current_length < char_count:
        # Create varied paragraph lengths
        para_length = random.randint(200, 800)
        sentences = []
        
        # Generate sentences for this paragraph
        while sum(len(s) for s in sentences) < para_length and current_length < char_count:
            # Create varied sentence templates
            templates = [
                "The analysis of {} indicates that {} is a critical factor in {}.",
                "Research conducted by {} demonstrates significant implications for {} in the context of {}.",
                "According to {}, the relationship between {} and {} requires further investigation.",
                "The implementation of {} has resulted in unexpected consequences for {} and {}.",
                "Despite concerns about {}, evidence suggests that {} remains effective for {}.",
                "Experts in {} continue to debate whether {} actually influences {}.",
                "The development of {} represents a major advancement in how we understand {} and {}.",
                "Historical data regarding {} contradicts current theories about {} in relation to {}.",
                "The impact of {} on {} has been significantly overestimated according to recent {} studies.",
                "Preliminary findings regarding {} suggest a correlation with {} under specific {} conditions."
            ]
            
            # Fill in templates with random terms
            terms = [
                "machine learning", "neural networks", "computer vision", "natural language processing",
                "quantum computing", "blockchain technology", "cloud infrastructure", "data privacy",
                "artificial intelligence", "systems architecture", "ethical considerations", "user experience",
                "algorithmic bias", "computational efficiency", "resource allocation", "security protocols",
                "network topology", "distributed systems", "parameter optimization", "data structures",
                "memory management", "compiler design", "operating systems", "virtualization",
                "database management", "information retrieval", "knowledge graphs", "semantic analysis"
            ]
            
            template = random.choice(templates)
            filled_template = template.format(
                random.choice(terms),
                random.choice(terms),
                random.choice(terms)
            )
            
            sentences.append(filled_template)
            current_length += len(filled_template) + 1  # +1 for space
        
        # Join sentences into a paragraph
        paragraph = " ".join(sentences)
        paragraphs.append(paragraph)
        
        # Early exit if we've exceeded the target
        if current_length >= char_count:
            break
    
    # Join paragraphs with double newlines
    text = "\n\n".join(paragraphs)
    
    # Trim to exact length if needed
    if len(text) > char_count:
        text = text[:char_count]
    
    return text

def insert_needle(haystack, needle, position_percentage=None):
    """
    Inserts the needle into a specific position within the haystack.
    
    Args:
        haystack: The large text to insert the needle into
        needle: The text to insert
        position_percentage: Optional, if provided will insert at approximately 
                            this percentage through the text (0-100)
    
    Returns:
        Tuple of (haystack_with_needle, actual_percentage)
    """
    # Determine insertion point
    if position_percentage is not None:
        # Convert percentage to position
        percentage = max(0, min(100, position_percentage)) / 100
        insertion_point = int(len(haystack) * percentage)
    else:
        # Random insertion between 10% and 90% of the document
        max_insertion_point = max(0, len(haystack) - len(needle) - 100)
        min_point = int(len(haystack) * 0.1)
        max_point = int(len(haystack) * 0.9)
        insertion_point = random.randint(min_point, max_point) if max_insertion_point > 100 else len(haystack) // 2

    # Calculate percentage position for reporting
    actual_percentage = (insertion_point / len(haystack)) * 100
    
    haystack_with_needle = haystack[:insertion_point] + "\n\n--- NEEDLE START --- \n" + needle + "\n--- NEEDLE END --- \n\n" + haystack[insertion_point:]
    
    return haystack_with_needle, actual_percentage

def create_prompt(text_with_needle, question):
    """
    Formats the final prompt for the LLM.
    
    Args:
        text_with_needle: The combined haystack and needle text
        question: The question to ask about the needle
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""Here is a very long document:

<document>
{text_with_needle}
</document>

Based *only* on the document provided above, please answer the following question:
Question: {question}
Answer:"""
    return prompt 