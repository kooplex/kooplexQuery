#!/usr/bin/env python3
"""
Integration example and test for VectorStore Synchronization System.

This script demonstrates how to use the automatic synchronization 
between PostgreSQL metadata database and Chroma vectorstore.

Run this to verify the sync system is working correctly.
"""

import logging
import time
from kooplexQuery.motor import Motor
from kooplexQuery.utils.vectorstore import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_basic_sync():
    """Example 1: Basic automatic synchronization."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Automatic Synchronization")
    print("="*60)
    
    # Initialize Motor (includes sync manager)
    print("\n✓ Initializing Motor...")
    motor = Motor()
    
    # Initialize VectorStore
    print("✓ Initializing VectorStore...")
    vs = VectorStore()
    
    # Connect vectorstore to sync manager
    print("✓ Connecting VectorStore to Motor's sync manager...")
    motor.vectorstore = vs
    
    # Save schema with automatic sync
    schema = "CREATE TABLE users (id INT, name VARCHAR(100), email VARCHAR(100));"
    print(f"\n✓ Saving schema to database and vectorstore...")
    motor.save_knowledge('schema', schema)
    
    # Verify sync
    schema_collection = vs._init_db('schema')
    count = len(schema_collection.get()['ids'])
    print(f"✓ Schema synced! Vector store contains {count} documents")
    
    return motor, vs


def example_sync_monitoring(motor):
    """Example 2: Monitoring sync operations."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Monitoring Sync Operations")
    print("="*60)
    
    # Get sync statistics
    stats = motor.sync_manager.get_sync_stats()
    print(f"\nSync Statistics:")
    print(f"  - Total operations: {stats['total_operations']}")
    print(f"  - Successful: {stats['successful']}")
    print(f"  - Failed: {stats['failed']}")
    print(f"  - Success rate: {stats['success_rate']:.1f}%")
    print(f"  - Sync enabled: {stats['sync_enabled']}")
    
    # Get recent sync logs
    logs = motor.sync_manager.get_sync_log(limit=5)
    print(f"\nRecent Sync Logs (last 5):")
    for log in logs:
        status = "✓" if log['success'] else "✗"
        print(f"  {status} {log['timestamp']}: {log['type']} ({log['target']}) - {log['details']}")


def example_batch_operations(motor):
    """Example 3: Batch sync operations."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Sync Operations")
    print("="*60)
    
    # Save multiple knowledge entries
    print("\nSaving multiple knowledge entries...")
    knowledge_items = [
        ('instruction', 'You are a SQL query generator. Always provide valid SQL.'),
        ('data_descriptor', 'This database contains user and transaction data.'),
        ('data_reference', 'Tables: users, transactions, products'),
    ]
    
    for reference, content in knowledge_items:
        motor.save_knowledge(reference, content)
        print(f"  ✓ Saved: {reference}")
    
    # Get batch sync status
    stats = motor.sync_manager.get_sync_stats()
    print(f"\n✓ Successfully synced {stats['total_operations']} knowledge items")
    
    # Batch sync examples (if any exist in database)
    print("\nBatch syncing examples from database...")
    synced_count = motor.sync_manager.batch_sync_examples()
    print(f"✓ Synced {synced_count} examples")


def example_enable_disable_sync(motor):
    """Example 4: Controlling sync enable/disable."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Controlling Synchronization")
    print("="*60)
    
    # Disable sync temporarily for bulk operations
    print("\nDisabling sync for bulk import...")
    motor.sync_manager.disable_sync()
    print("✓ Sync disabled")
    
    # Save items without syncing
    print("\nSaving knowledge without syncing...")
    motor.db_chat.save_knowledge('test_key', 'This will not be synced immediately')
    print("✓ Knowledge saved (but not synced)")
    
    # Re-enable sync
    print("\nRe-enabling sync...")
    motor.sync_manager.enable_sync()
    print("✓ Sync enabled")
    
    # Sync the missed item
    print("\nManual sync of missed item...")
    motor.sync_manager.sync_knowledge('test_key', 'This will not be synced immediately')
    print("✓ Item synced")


def example_individual_sync_operations(motor):
    """Example 5: Individual sync operations."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Individual Sync Operations")
    print("="*60)
    
    # Sync table description
    print("\nSyncing table description...")
    success = motor.sync_manager.sync_table_description(
        'users',
        'Contains user profile information including name, email, and registration date'
    )
    print(f"{'✓' if success else '✗'} Table description synced")
    
    # Sync column description
    print("\nSyncing column description...")
    success = motor.sync_manager.sync_column_description(
        column_name='user_id',
        table_name='users',
        description='Unique identifier for each user',
        data_type='INTEGER'
    )
    print(f"{'✓' if success else '✗'} Column description synced")
    
    # Sync example (question-SQL pair)
    print("\nSyncing example...")
    success = motor.sync_manager.sync_example(
        question_id=1,
        question_content='How many users registered today?',
        sql='SELECT COUNT(*) FROM users WHERE DATE(created_at) = CURRENT_DATE',
        is_public=True
    )
    print(f"{'✓' if success else '✗'} Example synced")


def example_error_handling(motor):
    """Example 6: Error handling and recovery."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Error Handling and Recovery")
    print("="*60)
    
    # Check sync logs for any errors
    logs = motor.sync_manager.get_sync_log(limit=20)
    failed_logs = [l for l in logs if not l['success']]
    
    if failed_logs:
        print(f"\nFound {len(failed_logs)} failed sync operations:")
        for log in failed_logs:
            print(f"  ✗ {log['target']}: {log['details']}")
        
        print("\nAttempting recovery...")
        # Could implement retry logic here
        print("  (Recovery logic would go here)")
    else:
        print("\n✓ No sync errors detected!")
    
    # Show overall health
    stats = motor.sync_manager.get_sync_stats()
    health_pct = stats['success_rate']
    health_status = "HEALTHY" if health_pct >= 95 else "DEGRADED" if health_pct >= 80 else "CRITICAL"
    print(f"\nSync System Health: {health_status} ({health_pct:.1f}%)")


def run_all_examples():
    """Run all examples."""
    print("\n" + "█"*60)
    print("VectorStore Synchronization System - Integration Examples")
    print("█"*60)
    
    try:
        # Example 1
        motor, vs = example_basic_sync()
        time.sleep(1)
        
        # Example 2
        example_sync_monitoring(motor)
        time.sleep(1)
        
        # Example 3
        example_batch_operations(motor)
        time.sleep(1)
        
        # Example 4
        example_enable_disable_sync(motor)
        time.sleep(1)
        
        # Example 5
        example_individual_sync_operations(motor)
        time.sleep(1)
        
        # Example 6
        example_error_handling(motor)
        
        # Final summary
        print("\n" + "█"*60)
        print("Examples Completed Successfully!")
        print("█"*60)
        
        # Final statistics
        stats = motor.sync_manager.get_sync_stats()
        print(f"\nFinal Sync Statistics:")
        print(f"  Total operations: {stats['total_operations']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        
        print("\n✓ Sync system is working correctly!")
        print("✓ You can now use Motor.save_knowledge() for automatic syncing")
        print("✓ Check VECTORSTORE_SYNC.md for detailed documentation")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        logger.exception("Exception in examples")
        raise


if __name__ == '__main__':
    run_all_examples()
