import os
import yake
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader
import torch

class EnhancedPaperAnalyzer:
    def __init__(self):
        # Initialize YAKE for keyword extraction
        self.kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=2,
            dedupLim=0.7,
            top=15,
            features=None
        )
        
        # Initialize T5 for summarization
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def extract_text_from_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def extract_keywords(self, text):
        keywords = self.kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]  # Return only keywords, not scores

    def generate_summary(self, text, max_length=150):
        prefix = "summarize: "
        input_text = prefix + text.strip()
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)
        
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def process_paper(self, pdf_path):
        text = self.extract_text_from_pdf(pdf_path)
        
        # Split into sections (improved section detection)
        sections = {
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'results': '',
            'discussion': '',
            'conclusion': ''
        }
        
        current_section = None
        lines = text.lower().split('\n')
        
        for line in lines:
            line = line.strip()
            for section in sections.keys():
                if section in line and len(line) < 50:
                    current_section = section
                    break
            if current_section and line:
                sections[current_section] += line + ' '

        # Process each section
        results = {
            'keywords': self.extract_keywords(text),
            'sections': {}
        }
        
        for section, content in sections.items():
            if len(content.strip()) > 100:
                results['sections'][section] = {
                    'summary': self.generate_summary(content),
                    'keywords': self.extract_keywords(content)
                }

        return results

def main():
    analyzer = EnhancedPaperAnalyzer()
    articles_dir = '../articles'
    output_dir = '../output'
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(articles_dir):
        if filename.endswith('.pdf'):
            print(f"\nProcessing {filename}...")
            pdf_path = os.path.join(articles_dir, filename)
            results = analyzer.process_paper(pdf_path)
            
            # Save results in a well-formatted text file
            output_file = os.path.join(output_dir, f'{filename[:-4]}_analysis.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Analysis Report for: {filename}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("DOCUMENT KEYWORDS:\n")
                f.write("-" * 20 + "\n")
                for kw in results['keywords']:
                    f.write(f"• {kw}\n")
                f.write("\n")
                
                f.write("SECTION-WISE ANALYSIS:\n")
                f.write("-" * 20 + "\n")
                
                for section, content in results['sections'].items():
                    if content:
                        f.write(f"\n{section.upper()}:\n")
                        f.write("~" * len(section) + "\n")
                        f.write("\nKey Points:\n")
                        for kw in content['keywords'][:5]:
                            f.write(f"• {kw}\n")
                        f.write("\nSummary:\n")
                        f.write(content['summary'])
                        f.write("\n")
                        f.write("-" * 50 + "\n")
            
            print(f"Analysis completed for {filename}")

if __name__ == "__main__":
    main()