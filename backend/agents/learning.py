"""
Learning Agent for Astraeus

This agent specializes in educational features, providing tools for:
- Document analysis and concept extraction
- Study guide generation
- Quiz and assessment creation
- Learning path recommendations
"""

from typing import Dict, List, Optional, Any
import logging
from ..utils.embeddings import EmbeddingManager
from ..utils.conversation import ConversationManager
from ..config import GOOGLE_API_KEY, LLM_MODEL
import google.generativeai as genai
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningAgent:
    """
    A specialized agent for educational features and learning assistance.
    
    This agent builds upon the DataIntakeAgent's capabilities to provide
    educational features like study guides, quizzes, and learning paths.
    
    Design Pattern: Strategy Pattern
    - Uses different strategies for different learning tasks
    - Easily extensible for new learning features
    - Maintains separation of concerns
    
    Key Components:
    1. Document Analysis: Extracts key concepts and learning objectives
    2. Content Generation: Creates study materials and assessments
    3. Progress Tracking: Monitors learning progress and adapts content
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize the Learning Agent.
        
        Args:
            embedding_manager: Reference to the shared embedding manager
                             for accessing document embeddings and content
                             
        Design Decision: Dependency Injection
        - Instead of creating its own embedding manager, we inject it
        - This promotes loose coupling and reusability
        - Allows sharing the same document store across agents
        """
        self.embedding_manager = embedding_manager
        self.conversation_manager = ConversationManager()
        
        # Initialize the LLM for content generation
        if GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                self.llm = genai.GenerativeModel(LLM_MODEL)
                logger.info("Learning Agent LLM configured successfully")
            except Exception as e:
                logger.error(f"Failed to configure LLM: {e}")
                self.llm = None
        else:
            logger.warning("GOOGLE_API_KEY not found. Some features may be limited.")
            self.llm = None
            
    def analyze_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Analyze a document to extract key learning concepts and structure.
        
        Args:
            doc_id: The ID of the document to analyze
            
        Returns:
            Dict containing:
            - key_concepts: List of main concepts
            - learning_objectives: List of learning goals
            - difficulty_level: Estimated complexity
            - prerequisite_concepts: Related concepts to understand first
            
        Implementation Strategy:
        1. Use embeddings to find key concepts
        2. Generate learning objectives using LLM
        3. Analyze text complexity
        4. Identify prerequisite knowledge
        """
        try:
            # Get document content from embedding manager
            doc_info = self.embedding_manager.documents.get(doc_id)
            if not doc_info:
                raise ValueError(f"Document {doc_id} not found")
                
            content = doc_info.get('content', '')
            metadata = doc_info.get('metadata', {})
            
            if not self.llm:
                raise ValueError("LLM not configured. Cannot perform document analysis.")
                
            # Step 1: Extract key concepts using LLM
            concept_prompt = f"""
            Analyze the following document content and extract the key concepts.
            Focus on main ideas, important terms, and core principles.
            Format the response as a JSON with a 'concepts' array.
            
            Content:
            {content[:2000]}  # Limit content length for prompt
            
            Response format:
            {{
                "concepts": [
                    {{
                        "name": "concept name",
                        "description": "brief explanation",
                        "importance": "high/medium/low"
                    }}
                ]
            }}
            """
            
            concept_response = self.llm.generate_content(concept_prompt)
            key_concepts = self._parse_llm_response(concept_response.text, 'concepts')
            
            # Step 2: Generate learning objectives
            objectives_prompt = f"""
            Based on the following document content, generate clear learning objectives.
            Focus on what a learner should be able to understand or do after studying this material.
            
            Content:
            {content[:2000]}
            
            Response format:
            {{
                "objectives": [
                    "clear, actionable learning objective",
                    "another learning objective"
                ]
            }}
            """
            
            objectives_response = self.llm.generate_content(objectives_prompt)
            learning_objectives = self._parse_llm_response(objectives_response.text, 'objectives')
            
            # Step 3: Analyze difficulty level
            difficulty_prompt = f"""
            Analyze the complexity of this document and determine its difficulty level.
            Consider factors like:
            - Technical terminology
            - Conceptual complexity
            - Required background knowledge
            
            Content:
            {content[:2000]}
            
            Response format:
            {{
                "difficulty": "beginner/intermediate/advanced",
                "reasoning": "explanation of the difficulty assessment"
            }}
            """
            
            difficulty_response = self.llm.generate_content(difficulty_prompt)
            difficulty_info = self._parse_llm_response(difficulty_response.text)
            
            # Step 4: Identify prerequisites
            prereq_prompt = f"""
            Identify prerequisite knowledge or concepts needed to understand this document.
            Consider both specific topics and general background knowledge.
            
            Content:
            {content[:2000]}
            
            Response format:
            {{
                "prerequisites": [
                    {{
                        "topic": "prerequisite topic",
                        "importance": "essential/recommended",
                        "reason": "why this prerequisite is needed"
                    }}
                ]
            }}
            """
            
            prereq_response = self.llm.generate_content(prereq_prompt)
            prerequisites = self._parse_llm_response(prereq_response.text, 'prerequisites')
            
            return {
                'status': 'success',
                'doc_id': doc_id,
                'analysis': {
                    'key_concepts': key_concepts,
                    'learning_objectives': learning_objectives,
                    'difficulty_level': difficulty_info.get('difficulty', 'intermediate'),
                    'difficulty_reasoning': difficulty_info.get('reasoning', ''),
                    'prerequisite_concepts': prerequisites
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document {doc_id}: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
            
    def _parse_llm_response(self, response_text: str, key: str = None) -> Any:
        """
        Parse the LLM response text into structured data.
        
        Args:
            response_text: The raw text response from the LLM
            key: Optional specific key to extract from the response
            
        Returns:
            Parsed data structure or specific value if key is provided
        """
        try:
            # Find JSON content in the response
            import json
            import re
            
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in LLM response")
                
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            return data[key] if key else data
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return [] if key else {}
            
    def generate_study_guide(self, doc_id: str, style: str = 'comprehensive') -> Dict[str, Any]:
        """
        Generate a study guide from a document.
        
        Args:
            doc_id: The ID of the document to create a guide for
            style: The style of guide ('comprehensive', 'quick', 'detailed')
            
        Returns:
            Dict containing the study guide content and metadata
            
        Implementation Strategy:
        1. Use document analysis results
        2. Generate structured content using LLM
        3. Format according to requested style
        """
        try:
            # First analyze the document
            analysis = self.analyze_document(doc_id)
            if analysis['status'] != 'success':
                raise ValueError(f"Failed to analyze document: {analysis.get('message')}")
                
            # Get document content
            doc_info = self.embedding_manager.documents.get(doc_id)
            content = doc_info.get('content', '')
            
            # Generate study guide based on style
            if style == 'quick':
                guide_prompt = f"""
                Create a quick study guide for the following document.
                Focus on key points and essential information.
                
                Document Analysis:
                {analysis['analysis']}
                
                Content:
                {content[:2000]}
                
                Response format:
                {{
                    "overview": "brief overview of the topic",
                    "key_points": [
                        "main point 1",
                        "main point 2"
                    ],
                    "summary": "concise summary of the content"
                }}
                """
            elif style == 'detailed':
                guide_prompt = f"""
                Create a detailed study guide for the following document.
                Include comprehensive explanations and examples.
                
                Document Analysis:
                {analysis['analysis']}
                
                Content:
                {content[:2000]}
                
                Response format:
                {{
                    "overview": "detailed overview of the topic",
                    "sections": [
                        {{
                            "title": "section title",
                            "content": "detailed content",
                            "examples": ["example 1", "example 2"]
                        }}
                    ],
                    "key_concepts": [
                        {{
                            "concept": "concept name",
                            "explanation": "detailed explanation",
                            "application": "how it's used"
                        }}
                    ],
                    "practice_questions": [
                        {{
                            "question": "practice question",
                            "answer": "answer explanation"
                        }}
                    ]
                }}
                """
            else:  # comprehensive (default)
                guide_prompt = f"""
                Create a comprehensive study guide for the following document.
                Include a balance of overview, details, and practical applications.
                
                Document Analysis:
                {analysis['analysis']}
                
                Content:
                {content[:2000]}
                
                Response format:
                {{
                    "overview": "comprehensive overview of the topic",
                    "learning_objectives": [
                        "objective 1",
                        "objective 2"
                    ],
                    "main_sections": [
                        {{
                            "title": "section title",
                            "content": "main content",
                            "key_points": ["point 1", "point 2"],
                            "examples": ["example 1", "example 2"]
                        }}
                    ],
                    "key_concepts": [
                        {{
                            "concept": "concept name",
                            "explanation": "clear explanation",
                            "importance": "why it matters"
                        }}
                    ],
                    "practice_questions": [
                        {{
                            "question": "practice question",
                            "answer": "answer explanation",
                            "difficulty": "easy/medium/hard"
                        }}
                    ],
                    "additional_resources": [
                        "suggested reading or practice materials"
                    ]
                }}
                """
            
            # Generate the study guide content
            guide_response = self.llm.generate_content(guide_prompt)
            guide_content = self._parse_llm_response(guide_response.text)
            
            # Add metadata and style information
            guide_content.update({
                'metadata': {
                    'doc_id': doc_id,
                    'style': style,
                    'generated_at': datetime.datetime.now().isoformat(),
                    'difficulty_level': analysis['analysis']['difficulty_level']
                }
            })
            
            return {
                'status': 'success',
                'doc_id': doc_id,
                'style': style,
                'content': guide_content
            }
            
        except Exception as e:
            logger.error(f"Error generating study guide for {doc_id}: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            } 