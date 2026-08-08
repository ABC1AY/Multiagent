"""Generate a small sample TriviaQA-style dataset for testing.

Use this when you can't download the real TriviaQA dataset due to network issues.
"""
import json
from pathlib import Path

import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import DATA_PROCESSED_DIR


SAMPLE_DATA = [
    {
        "id": "sample_001",
        "document": """== Python (programming language) ==
Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python was conceived in the late 1980s by Guido van Rossum and released in 1991.

== History ==
Python was created by Guido van Rossum, who began implementing it in December 1989 as a successor to the ABC programming language. Van Rossum is Python's principal author and his continuing central role in deciding the direction of Python is reflected in the title given to him by the Python community, benevolent dictator for life.

== Features ==
Python features a dynamic type system and automatic memory management. It has a large and comprehensive standard library.""",
        "question": "Who created the Python programming language?",
        "answer": "Guido van Rossum",
        "aliases": ["Guido van Rossum", "van Rossum", "Guido"],
    },
    {
        "id": "sample_002",
        "document": """== Paris ==
Paris is the capital and most populous city of France. Situated on the Seine River, in the north of the country, it is in the centre of the Île-de-France region, also known as the région parisienne.

== History ==
Paris was founded in the 3rd century BC by a Celtic people called the Parisii. By the 12th century, Paris was the largest city in the western world, a prosperous trading centre.

== Landmarks ==
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair.""",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "aliases": ["Paris", "City of Light"],
    },
    {
        "id": "sample_003",
        "document": """== Albert Einstein ==
Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science.

== Life and career ==
Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879. His parents were Hermann Einstein, a salesman and engineer, and Pauline Koch.

== Achievements ==
He received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect.""",
        "question": "When was Albert Einstein born?",
        "answer": "14 March 1879",
        "aliases": ["March 14, 1879", "1879-03-14"],
    },
    {
        "id": "sample_004",
        "document": """== World War II ==
World War II or the Second World War was a global conflict that lasted from 1939 to 1945. The vast majority of the world's countries fought in it, forming two opposing military alliances: the Allies and the Axis powers.

== Course of the war ==
The war began with the German invasion of Poland on 1 September 1939. Britain and France declared war on Germany two days later. The war ended with the unconditional surrender of the Axis powers in 1945.

== Impact ==
World War II was the deadliest conflict in human history, causing the deaths of 70 to 85 million people.""",
        "question": "When did World War II begin?",
        "answer": "1 September 1939",
        "aliases": ["September 1, 1939", "1939-09-01"],
    },
    {
        "id": "sample_005",
        "document": """== DNA ==
Deoxyribonucleic acid is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses.

== Structure ==
The two DNA strands are known as polynucleotides as they are composed of simpler monomeric units called nucleotides. Each nucleotide is composed of one of four nitrogen-containing nucleobases, a sugar called deoxyribose, and a phosphate group.

== Discovery ==
The structure of DNA was discovered by James Watson and Francis Crick in 1953, with crucial contributions from Rosalind Franklin and Maurice Wilkins.""",
        "question": "Who discovered the structure of DNA?",
        "answer": "James Watson and Francis Crick",
        "aliases": ["Watson and Crick", "Watson", "Crick"],
    },
]


def generate_sample_dataset(output_path: Path = DATA_PROCESSED_DIR / "triviaqa_sample.jsonl"):
    """Write sample TriviaQA-style data to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in SAMPLE_DATA:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {len(SAMPLE_DATA)} sample samples -> {output_path}")
    return SAMPLE_DATA


if __name__ == "__main__":
    generate_sample_dataset()
