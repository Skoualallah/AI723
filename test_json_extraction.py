#!/usr/bin/env python3
"""Test script for JSON extraction functionality"""

from json_utils import extract_json_from_text, extract_answer_letter_from_json

# Test case 1: JSON with markdown code block
test1 = """```json
{
 "explanation": "I am an artificial intelligence, a large language model. In common understanding, when someone asks if I am a robot, they are typically trying to determine if they are interacting with a human or an automated system/AI. As an AI, I am an automated system designed to perform tasks, which aligns with the broader definition of a 'robot' or 'bot' in the digital sense, as opposed to a human. Therefore, 'yes' is the most appropriate answer to indicate my nature as a non-human, automated entity.",
 "final_answer": "oui",
 "final_answer_letter": "a"
}
```"""

# Test case 2: JSON with text before and after
test2 = """Here is my response in JSON format:

{
 "explanation": "I am an artificial intelligence, a large language model. In common understanding, when someone asks if I am a robot, they are typically trying to determine if they are interacting with a human or an automated system/AI. As an AI, I am an automated system designed to perform tasks, which aligns with the broader definition of a 'robot' or 'bot' in the digital sense, as opposed to a human. Therefore, 'yes' is the most appropriate answer to indicate my nature as a non-human, automated entity.",
 "final_answer": "oui",
 "final_answer_letter": "a"
}

I hope this helps!"""

# Test case 3: Pure JSON
test3 = """{
 "explanation": "I am an artificial intelligence, a large language model. In common understanding, when someone asks if I am a robot, they are typically trying to determine if they are interacting with a human or an automated system/AI. As an AI, I am an automated system designed to perform tasks, which aligns with the broader definition of a 'robot' or 'bot' in the digital sense, as opposed to a human. Therefore, 'yes' is the most appropriate answer to indicate my nature as a non-human, automated entity.",
 "final_answer": "oui",
 "final_answer_letter": "a"
}"""

# Test case 4: JSON with extra text in the middle (should still work)
test4 = """Sure, let me answer that.

{
 "explanation": "Test explanation",
 "final_answer": "test",
 "final_answer_letter": "B"
}

This is the end."""

def test_extraction(test_name, text):
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")

    result = extract_json_from_text(text)

    if result:
        print("✅ JSON extracted successfully!")
        print(f"Explanation: {result.get('explanation', 'N/A')[:100]}...")
        print(f"Final Answer: {result.get('final_answer', 'N/A')}")
        print(f"Final Answer Letter: {result.get('final_answer_letter', 'N/A')}")

        letter = extract_answer_letter_from_json(result)
        print(f"Extracted Letter (uppercase): {letter}")
    else:
        print("❌ Failed to extract JSON")

# Run all tests
test_extraction("JSON in markdown code block", test1)
test_extraction("JSON with text before and after", test2)
test_extraction("Pure JSON", test3)
test_extraction("JSON with surrounding text", test4)

print(f"\n{'='*60}")
print("All tests completed!")
print(f"{'='*60}\n")
