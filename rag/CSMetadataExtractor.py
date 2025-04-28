import nltk
from nltk.corpus import stopwords
from NER_Extractor import NER_Extractor
STOPWORDS_EN = set(stopwords.words('english'))
STOPWORDS_RO = set(stopwords.words('romanian'))
import json
from collections import Counter
from datetime import datetime
from typing import Dict, Any, List
import re

class CSMetadataExtractor:

    def __init__(self, keywords_file="cs_keywords.json", eras_file="cs_eras.json"):
        # Pattern for detecting years
        self.year_pattern = re.compile(r'\b(1[0-9]{3}|20[0-2][0-9])\b')
        # Pattern for identifying section titles
        self.section_pattern = re.compile(r'={2,}(.*?)={2,}')

        # Load CS domain keywords from file
        self.cs_keywords = self._load_keywords(keywords_file)

        # Load CS eras from file
        self.cs_eras = self._load_eras(eras_file)

        self.all_cs_keywords = set()
        for category, keywords in self.cs_keywords.items():
            self.all_cs_keywords.update([k.lower() for k in keywords])

        self.code_pattern = re.compile(r'`([^`]+)`|```[\s\S]*?```')
        self.version_pattern = re.compile(r'\b\d+\.\d+(?:\.\d+)?\b')
        self.technical_pattern = re.compile(r'\b[A-Z][A-Za-z0-9]*(?:\.[A-Za-z0-9]+)+\b')

    def _load_keywords(self, file_path: str) -> Dict[str, List[str]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading keywords file: {e}")
            # Provide a minimal fallback set of keywords
            return {
                "programming_languages": ["Python", "Java", "C++", "JavaScript"],
                "romanian_terms": ["algoritm", "programare", "funcție", "clasă"]
            }

    def _load_eras(self, file_path: str) -> List[Dict[str, Any]]:

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading eras file: {e}")
            return [
                {"name": "Early Computing", "start_year": 1940, "end_year": 1959},
                {"name": "Modern Computing", "start_year": 1960, "end_year": 2030}
            ]

    def extract_metadata(self, text: str, title: str, lang: str) -> Dict[str, Any]:
        years = self._extract_years(text)

        # Extract CS-specific keywords
        cs_topics = self._extract_cs_topics(text, lang)

        # Extract general keywords
        keywords = self._extract_keywords(text, lang)

        # Extract technical details (code snippets, versions, methods, etc.)
        technical_details = self._extract_technical_details(text, lang)

        # Identify sections in text
        sections = self._extract_sections(text)

        # Estimate time period covered and CS era
        time_period = self._estimate_cs_time_period(years)

        # Determine content type
        content_type = self._determine_cs_content_type(text, title, cs_topics)

        # Estimate reading time in minutes
        reading_time = self._estimate_reading_time(text, lang)

        # Determine complexity level
        complexity = self._estimate_cs_complexity(text, lang, technical_details)

        # Create and return metadata dictionary
        metadata = {
            "cs_topics": cs_topics,
            "keywords": keywords,
            "years_mentioned": years[:20] if len(years) > 20 else years,
            "technical_details": technical_details,
            "sections": sections[:10] if len(sections) > 10 else sections,
            "time_period": time_period,
            "content_type": content_type,
            "reading_time_minutes": reading_time,
            "complexity_level": complexity,
            "extraction_date": datetime.now().strftime("%Y-%m-%d")
        }

        return metadata

    def _extract_years(self, text: str) -> List[int]:
        """Extract years mentioned in text."""
        years = [int(y) for y in self.year_pattern.findall(text)]
        return sorted(list(set(years)))

    def _extract_cs_topics(self, text: str, lang: str) -> Dict[str, List[str]]:
        """
        Extract computer science specific topics and keywords.

        Returns a dictionary mapping topic categories to found keywords.
        """
        text_lower = text.lower()
        found_topics = {}

        # Check each category and its keywords
        for category, keywords in self.cs_keywords.items():
            found_keywords = []

            for keyword in keywords:
                # For multi-word keywords, check exact matches
                if ' ' in keyword and keyword.lower() in text_lower:
                    found_keywords.append(keyword)
                # For single words, check word boundaries
                elif ' ' not in keyword and re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower):
                    found_keywords.append(keyword)

            if found_keywords:
                found_topics[category] = found_keywords

        return found_topics

    def _extract_keywords(self, text: str, lang: str) -> List[str]:
        """Extract general keywords from text."""
        extractor = NER_Extractor()
        keywords = extractor.extract(text)
        keywords = [kw[0] for kw in keywords if kw[1] == "Keyword"]
        keywords = [kw for kw in keywords if kw.lower() not in self.all_cs_keywords]
        keywords = [kw for kw in keywords if kw.lower() not in STOPWORDS_EN]
        keywords = [kw for kw in keywords if len(kw) > 2]
        keywords = [kw for kw in keywords if kw not in self.cs_keywords]
        keywords = [kw for kw in keywords if kw not in self.cs_eras]
        return keywords

    def _extract_technical_details(self, text: str, lang: str) -> Dict[str, Any]:
        """Extract technical details like code snippets, versions, methods."""
        details = {}

        # Extract code snippets
        code_snippets = self.code_pattern.findall(text)
        if code_snippets:
            details["code_snippets_count"] = len(code_snippets)
            if code_snippets:
                sample = code_snippets[0]
                if len(sample) > 100:
                    sample = sample[:100] + "..."
                details["code_sample"] = sample

        versions = self.version_pattern.findall(text)
        if versions:
            details["versions_mentioned"] = versions[:5]

            # Extract technical methods/functions
        technical_methods = self.technical_pattern.findall(text)
        if technical_methods:
            details["technical_methods"] = technical_methods[:5]

            # Check for common file extensions
        file_extensions = re.findall(r'\b\w+\.(py|java|cpp|js|html|css|sql|json|xml|php|rb|go|rs)\b', text)
        if file_extensions:
            extension_count = Counter(file_extensions)
            details["file_types"] = dict(extension_count)

        return details

    def _extract_sections(self, text: str) -> List[str]:
        """Extract sections from Wikipedia text."""
        section_titles = self.section_pattern.findall(text)
        # Clean section titles
        clean_titles = [title.strip() for title in section_titles]
        return clean_titles

    def _estimate_cs_time_period(self, years: List[int]) -> Dict[str, Any]:
        """Estimate the time period covered, with focus on CS eras."""
        if not years:
            return {"specified": False}

        min_year = min(years)
        max_year = max(years)

        # Basic period info
        period = {
            "min_year": min_year,
            "max_year": max_year,
            "span_years": max_year - min_year
        }

        cs_era = None
        for era in self.cs_eras:
            if min_year >= era["start_year"] and max_year <= era["end_year"]:
                cs_era = era["name"]
                break

        if cs_era:
            period["cs_era"] = cs_era
        else:
            # Identify multiple eras if spanning multiple periods
            spanning_eras = []
            for era in self.cs_eras:
                if (min_year <= era["end_year"] and max_year >= era["start_year"]):
                    spanning_eras.append(era["name"])

            if spanning_eras:
                period["spanning_eras"] = spanning_eras

        return period

    def _determine_cs_content_type(self, text: str, title: str, cs_topics: Dict[str, List[str]]) -> str:
        lower_text = text.lower()
        lower_title = title.lower()

        # Determine by title keywords
        if any(term in lower_title for term in ["tutorial", "guide", "how to", "introduction"]):
            return "Tutorial/Guide"

        if any(term in lower_title for term in ["review", "comparison", "versus", "vs"]):
            return "Technology Review"

        if any(term in lower_title for term in ["history", "evolution", "development of"]):
            return "CS History"

        if any(term in lower_title for term in ["concept", "theory", "principle"]):
            return "Theoretical CS"

        if any(term in lower_title for term in ["algorithm", "implementation", "code"]):
            return "Algorithm/Implementation"

        if any(term in lower_title for term in ["language", "programming", "syntax"]):
            return "Programming Language"

        if any(term in lower_title for term in ["framework", "library", "tool"]):
            return "CS Tool/Framework"

        if any(term in lower_title for term in ["problem", "challenge", "solution"]):
            return "Problem Solving"

        # Determine by topic coverage
        if "programming_languages" in cs_topics and len(cs_topics.get("programming_languages", [])) > 2:
            return "Programming Language"

        if "data_structures_algorithms" in cs_topics and len(cs_topics.get("data_structures_algorithms", [])) > 2:
            return "Algorithms & Data Structures"

        if "ai_ml" in cs_topics and len(cs_topics.get("ai_ml", [])) > 2:
            return "AI & Machine Learning"

        if "databases" in cs_topics and len(cs_topics.get("databases", [])) > 2:
            return "Database Systems"

        if "web_development" in cs_topics and len(cs_topics.get("web_development", [])) > 2:
            return "Web Development"

        if "software_engineering" in cs_topics and len(cs_topics.get("software_engineering", [])) > 2:
            return "Software Engineering"

        if "computer_systems" in cs_topics and len(cs_topics.get("computer_systems", [])) > 2:
            return "Computer Systems"

        if "networking" in cs_topics and len(cs_topics.get("networking", [])) > 2:
            return "Computer Networks"

        if "security" in cs_topics and len(cs_topics.get("security", [])) > 2:
            return "Cybersecurity"

        if "theoretical_cs" in cs_topics and len(cs_topics.get("theoretical_cs", [])) > 2:
            return "Theoretical Computer Science"

        # Default
        return "General Computer Science"

    def _estimate_reading_time(self, text: str, lang: str) -> int:
        """
        Estimate reading time in minutes.
        Technical CS content is read more slowly, around 150 words per minute.
        """
        words = nltk.word_tokenize(text)
        minutes = len(words) / 150
        return round(minutes)

    def _estimate_cs_complexity(self, text: str, lang: str, technical_details: Dict[str, Any]) -> str:
        """Estimate text complexity level, with CS-specific considerations."""
        # Basic text complexity analysis
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)

        # Average sentence length
        avg_sentence_length = len(words) / len(sentences) if sentences else 0

        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        # CS-specific complexity factors
        complexity_score = 0

        # Code snippets increase complexity
        if technical_details.get("code_snippets_count", 0) > 5:
            complexity_score += 2
        elif technical_details.get("code_snippets_count", 0) > 0:
            complexity_score += 1

        # Technical methods increase complexity
        if len(technical_details.get("technical_methods", [])) > 3:
            complexity_score += 1

        # Count CS jargon terms
        cs_jargon_count = sum(1 for word in words if word.lower() in self.all_cs_keywords)
        jargon_density = cs_jargon_count / len(words) if words else 0

        if jargon_density > 0.1:  # More than 10% of words are CS terms
            complexity_score += 2
        elif jargon_density > 0.05:  # More than 5% are CS terms
            complexity_score += 1

        # Final complexity determination
        if complexity_score >= 3 or avg_sentence_length > 25 or avg_word_length > 7:
            return "Advanced"
        elif complexity_score >= 1 or avg_sentence_length > 15 or avg_word_length > 5.5:
            return "Intermediate"
        else:
            return "Beginner"