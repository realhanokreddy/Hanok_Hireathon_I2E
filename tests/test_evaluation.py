"""
Test and verification script for the QA system.
Evaluates retrieval quality, citation accuracy, and answer relevance.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.qa_system import TechnicalQASystem
from src.config import get_config


logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating QA performance."""
    total_queries: int
    successful_answers: int
    citation_accuracy: float
    avg_confidence: float
    retrieval_recall: float
    category_performance: Dict[str, float]


class QAEvaluator:
    """Evaluate QA system performance."""
    
    def __init__(self, qa_system: TechnicalQASystem):
        """
        Initialize evaluator.
        
        Args:
            qa_system: QA system instance
        """
        self.qa_system = qa_system
    
    def evaluate_test_queries(self, test_queries_file: str) -> EvaluationMetrics:
        """
        Evaluate system on test queries.
        
        Args:
            test_queries_file: Path to test queries JSON file
            
        Returns:
            Evaluation metrics
        """
        # Load test queries
        with open(test_queries_file, 'r') as f:
            test_queries = json.load(f)
        
        logger.info(f"Evaluating {len(test_queries)} test queries")
        
        results = []
        category_scores = {}
        
        for test_query in test_queries:
            query_id = test_query['id']
            query = test_query['query']
            category = test_query['category']
            expected_sections = test_query.get('expected_sections', [])
            expected_terms = test_query.get('expected_terms', [])
            
            logger.info(f"Testing query {query_id}: {query}")
            
            # Get answer
            result = self.qa_system.ask(query, include_context=True)
            
            # Evaluate result
            evaluation = self._evaluate_single_result(
                result,
                expected_sections,
                expected_terms
            )
            
            evaluation['query_id'] = query_id
            evaluation['category'] = category
            results.append(evaluation)
            
            # Track category performance
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(evaluation['overall_score'])
        
        # Calculate aggregate metrics
        metrics = self._calculate_metrics(results, category_scores)
        
        return metrics, results
    
    def _evaluate_single_result(
        self,
        result: Dict[str, Any],
        expected_sections: List[str],
        expected_terms: List[str]
    ) -> Dict[str, Any]:
        """Evaluate a single QA result."""
        evaluation = {
            'answer_provided': len(result['answer']) > 0,
            'has_citations': len(result['citations']) > 0,
            'section_recall': 0.0,
            'term_coverage': 0.0,
            'confidence': result['confidence'],
            'overall_score': 0.0
        }
        
        # Check if expected sections were retrieved
        if expected_sections:
            retrieved_sections = set()
            for citation in result['citations']:
                section_num = citation['section_number']
                # Check if matches any expected section (prefix match)
                for expected in expected_sections:
                    if section_num.startswith(expected):
                        retrieved_sections.add(expected)
            
            evaluation['section_recall'] = len(retrieved_sections) / len(expected_sections) if expected_sections else 0.0
        else:
            evaluation['section_recall'] = 1.0  # No specific sections expected
        
        # Check if expected terms appear in answer
        if expected_terms:
            answer_lower = result['answer'].lower()
            found_terms = sum(1 for term in expected_terms if term.lower() in answer_lower)
            evaluation['term_coverage'] = found_terms / len(expected_terms)
        else:
            evaluation['term_coverage'] = 1.0
        
        # Calculate overall score
        weights = {
            'answer_provided': 0.2,
            'has_citations': 0.2,
            'section_recall': 0.3,
            'term_coverage': 0.2,
            'confidence': 0.1
        }
        
        evaluation['overall_score'] = sum(
            evaluation[metric] * weight
            for metric, weight in weights.items()
        )
        
        return evaluation
    
    def _calculate_metrics(
        self,
        results: List[Dict],
        category_scores: Dict[str, List[float]]
    ) -> EvaluationMetrics:
        """Calculate aggregate metrics."""
        total = len(results)
        
        successful = sum(1 for r in results if r['answer_provided'])
        
        avg_citation_accuracy = sum(r['section_recall'] for r in results) / total if total > 0 else 0.0
        avg_confidence = sum(r['confidence'] for r in results) / total if total > 0 else 0.0
        avg_retrieval_recall = sum(r['section_recall'] for r in results) / total if total > 0 else 0.0
        
        category_performance = {
            cat: sum(scores) / len(scores) if scores else 0.0
            for cat, scores in category_scores.items()
        }
        
        return EvaluationMetrics(
            total_queries=total,
            successful_answers=successful,
            citation_accuracy=avg_citation_accuracy,
            avg_confidence=avg_confidence,
            retrieval_recall=avg_retrieval_recall,
            category_performance=category_performance
        )
    
    def print_evaluation_report(
        self,
        metrics: EvaluationMetrics,
        results: List[Dict]
    ):
        """Print evaluation report."""
        print("\n" + "=" * 80)
        print("QA SYSTEM EVALUATION REPORT")
        print("=" * 80 + "\n")
        
        print(f"Total Queries: {metrics.total_queries}")
        print(f"Successful Answers: {metrics.successful_answers}/{metrics.total_queries} "
              f"({metrics.successful_answers/metrics.total_queries*100:.1f}%)")
        print(f"Average Citation Accuracy: {metrics.citation_accuracy:.2%}")
        print(f"Average Confidence: {metrics.avg_confidence:.2%}")
        print(f"Average Retrieval Recall: {metrics.retrieval_recall:.2%}")
        
        print("\n--- Performance by Category ---")
        for category, score in sorted(metrics.category_performance.items()):
            print(f"  {category:25s}: {score:.2%}")
        
        print("\n--- Detailed Results ---")
        for result in results:
            status = "✓" if result['overall_score'] > 0.7 else "⚠" if result['overall_score'] > 0.5 else "✗"
            print(f"{status} Query {result['query_id']:2d} ({result['category']:20s}): "
                  f"Score={result['overall_score']:.2f}, "
                  f"Section Recall={result['section_recall']:.2f}, "
                  f"Term Coverage={result['term_coverage']:.2f}")
        
        print("\n" + "=" * 80)


def main():
    """Main function for running evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate QA system')
    parser.add_argument('--test-file', default='tests/test_queries.json', help='Test queries file')
    parser.add_argument('--output', help='Output file for detailed results (JSON)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize QA system
    print("Initializing QA system...")
    config = get_config()
    qa_system = TechnicalQASystem(config=config)
    print("✓ QA system initialized\n")
    
    # Create evaluator
    evaluator = QAEvaluator(qa_system)
    
    # Run evaluation
    print(f"Running evaluation on {args.test_file}...")
    metrics, results = evaluator.evaluate_test_queries(args.test_file)
    
    # Print report
    evaluator.print_evaluation_report(metrics, results)
    
    # Save detailed results
    if args.output:
        output_data = {
            'metrics': {
                'total_queries': metrics.total_queries,
                'successful_answers': metrics.successful_answers,
                'citation_accuracy': metrics.citation_accuracy,
                'avg_confidence': metrics.avg_confidence,
                'retrieval_recall': metrics.retrieval_recall,
                'category_performance': metrics.category_performance
            },
            'results': results
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Detailed results saved to {args.output}")


if __name__ == '__main__':
    main()
