"""
Re-ranking utilities for semantic search.

This module provides tools to re-rank search results for improved relevance.
"""
from typing import List, Dict, Any, Optional, Callable
import logging
import google.generativeai as genai
import re

# Set up logging
logger = logging.getLogger(__name__)

class SearchReranker:
    """
    Re-ranks search results to improve relevance.
    
    Uses various strategies to re-score and re-order search results:
    1. Cross-encoder scoring for better relevance
    2. Keyword matching for exact matches
    3. Chunk prioritization strategies
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the search reranker.
        
        Args:
            api_key: Optional Google API key for Gemini model
        """
        self.api_key = api_key
        
    def rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        strategy: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Re-rank search results using the specified strategy.
        
        Args:
            query: The original search query
            results: List of search results to re-rank
            strategy: Reranking strategy (hybrid, keyword, llm)
            
        Returns:
            Re-ranked list of search results
        """
        if not results:
            return []
            
        # Apply the selected strategy
        if strategy == "keyword":
            return self._keyword_rerank(query, results)
        elif strategy == "llm":
            return self._llm_rerank(query, results)
        else:  # "hybrid" (default)
            return self._hybrid_rerank(query, results)
    
    def _keyword_rerank(
        self, 
        query: str, 
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results based on keyword matching.
        
        Args:
            query: The original search query
            results: List of search results to re-rank
            
        Returns:
            Re-ranked list of search results
        """
        # Extract important keywords from the query
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return results  # No keywords to match, return original results
            
        # Score each result based on keyword presence
        scored_results = []
        for result in results:
            content = result.get("content", "").lower()
            
            # Count keyword matches
            keyword_score = sum(1 for keyword in keywords if keyword in content)
            
            # Combine with original score (weighted)
            original_score = result.get("score", 0)
            combined_score = (original_score * 0.7) + (keyword_score * 0.3 / len(keywords))
            
            # Update the result with new score
            result_copy = result.copy()
            result_copy["score"] = min(combined_score, 1.0)  # Cap at 1.0
            result_copy["keyword_matches"] = keyword_score
            
            scored_results.append(result_copy)
            
        # Sort by new score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results
    
    def _llm_rerank(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results using LLM relevance judgment.
        
        Args:
            query: The original search query
            results: List of search results to re-rank
            max_results: Maximum number of results to process with the LLM
            
        Returns:
            Re-ranked list of search results
        """
        if not self.api_key or not results:
            return results
            
        # Limit to top N results to reduce API costs
        top_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:max_results]
        
        try:
            # Prepare prompt for LLM
            prompt = self._build_reranking_prompt(query, top_results)
            
            # Call Gemini model
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            
            # Parse the response to get the new ranking
            new_order = self._parse_llm_response(response.text, [r.get("id") for r in top_results])
            
            if new_order and len(new_order) > 0:
                # Create a mapping from ID to result
                result_map = {r.get("id"): r for r in results}
                
                # Create the new ordered list based on LLM's ranking
                reranked = []
                for doc_id in new_order:
                    if doc_id in result_map:
                        result = result_map[doc_id].copy()
                        result["llm_ranked"] = True
                        reranked.append(result)
                
                # Add any remaining results that weren't ranked by the LLM
                ranked_ids = set(new_order)
                for result in results:
                    if result.get("id") not in ranked_ids:
                        reranked.append(result)
                
                return reranked
                
        except Exception as e:
            logger.error(f"Error in LLM reranking: {str(e)}")
            
        # Fallback to original results if anything fails
        return results
    
    def _hybrid_rerank(
        self, 
        query: str, 
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Re-rank using a hybrid of keyword and embedding approaches.
        
        Args:
            query: The original search query
            results: List of search results to re-rank
            
        Returns:
            Re-ranked list of search results
        """
        # First apply keyword reranking
        keyword_results = self._keyword_rerank(query, results)
        
        # Attempt LLM reranking on the top results if API key is available
        if self.api_key:
            try:
                # Only send top 3 results to LLM to reduce API costs
                top_results = keyword_results[:3]
                return self._llm_rerank(query, top_results) + keyword_results[3:]
            except Exception as e:
                logger.error(f"Error in hybrid reranking: {str(e)}")
                
        # Return keyword results as fallback
        return keyword_results
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from a query.
        
        Args:
            query: The search query
            
        Returns:
            List of keyword strings
        """
        # Remove common stop words
        stop_words = {"a", "an", "the", "and", "or", "but", "is", "are", "was", 
                      "were", "be", "have", "has", "had", "do", "does", "did",
                      "in", "on", "at", "to", "for", "with", "by", "about", "of"}
                      
        # Tokenize and filter query
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _build_reranking_prompt(
        self, 
        query: str, 
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Build a prompt for LLM-based reranking.
        
        Args:
            query: The original search query
            results: List of search results to re-rank
            
        Returns:
            Prompt string for the LLM
        """
        prompt = f"""I need you to re-rank these document chunks based on their relevance to the query: "{query}"

Here are the document chunks to rank:

"""
        # Add each result to the prompt
        for i, result in enumerate(results, 1):
            doc_id = result.get("id", f"doc_{i}")
            content = result.get("content", "")
            prompt += f"--- DOCUMENT {i} (ID: {doc_id}) ---\n{content}\n\n"
            
        prompt += """Please analyze each document chunk and rank them from most relevant to least relevant
considering exact matches, semantic relevance, and completeness of information.

Your response should be formatted as a JSON array of document IDs in order of relevance, like this:
["doc_id1", "doc_id2", "doc_id3"]

Only include the JSON array in your response, nothing else."""
        
        return prompt
    
    def _parse_llm_response(self, response: str, fallback_ids: List[str]) -> List[str]:
        """
        Parse the LLM's response to extract document IDs.
        
        Args:
            response: The text response from the LLM
            fallback_ids: IDs to use if parsing fails
            
        Returns:
            Ordered list of document IDs
        """
        try:
            # Look for JSON array pattern
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                import json
                doc_ids = json.loads(json_str)
                if isinstance(doc_ids, list) and all(isinstance(id, str) for id in doc_ids):
                    return doc_ids
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            
        return fallback_ids 