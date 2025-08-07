#!/usr/bin/env python3
"""
Fix for Geography Analysis in olist_ecommerce_analysis.ipynb
===========================================================

This script provides the corrected code for the geography analysis section
that's causing the KeyError: 'customer_state' issue.

Run this after running the earlier cells in the notebook to get the geographic analysis.
"""

import pandas as pd
import numpy as np

def fix_geography_analysis():
    """
    Fixed version of the geography analysis that handles the merge properly
    """
    print("🔧 Running Fixed Geography Analysis...")
    
    # Load the data (assuming you've already run the earlier cells)
    try:
        # These should be available from earlier cells
        orders = pd.read_csv('Data/olist_orders_dataset.csv')
        order_items = pd.read_csv('Data/olist_order_items_dataset.csv') 
        customers = pd.read_csv('Data/olist_customers_dataset.csv')
        payments = pd.read_csv('Data/olist_order_payments_dataset.csv')
        
        # Recreate customer_orders with proper merge
        print("📊 Recreating customer_orders dataframe...")
        
        # Step 1: Merge orders with order_items
        customer_orders = orders.merge(order_items, on='order_id', how='inner')
        print(f"After orders + order_items merge: {customer_orders.shape}")
        
        # Step 2: Add payment information
        payments_agg = payments.groupby('order_id')['payment_value'].sum().reset_index()
        customer_orders = customer_orders.merge(payments_agg, on='order_id', how='left')
        print(f"After payments merge: {customer_orders.shape}")
        
        # Step 3: Add customer information (THIS IS THE KEY STEP)
        print("🔗 Merging with customers data...")
        print(f"Customers columns: {customers.columns.tolist()}")
        print(f"Customer_orders columns before customer merge: {customer_orders.columns.tolist()}")
        
        # Perform the merge and check for customer_state
        customer_orders = customer_orders.merge(customers, on='customer_id', how='left')
        print(f"After customers merge: {customer_orders.shape}")
        print(f"Columns after merge: {customer_orders.columns.tolist()}")
        
        # Check if customer_state exists
        if 'customer_state' not in customer_orders.columns:
            print("❌ ERROR: customer_state still missing after merge!")
            print("Available columns:", customer_orders.columns.tolist())
            return None
        
        print("✅ customer_state column found!")
        
        # Now run the geographic analysis
        print("\n📈 Running Geographic Analysis...")
        
        geo_analysis = customer_orders  # Already merged above
        
        state_summary = geo_analysis.groupby('customer_state').agg({
            'customer_id': 'nunique',
            'order_id': 'nunique', 
            'payment_value': ['sum', 'mean'],
            'order_purchase_timestamp': ['min', 'max']
        }).round(2)
        
        state_summary.columns = ['unique_customers', 'total_orders', 'total_revenue', 'avg_order_value', 'first_order', 'last_order']
        state_summary['orders_per_customer'] = (state_summary['total_orders'] / state_summary['unique_customers']).round(2)
        state_summary = state_summary.sort_values('total_revenue', ascending=False)
        
        print("Geographic Distribution Analysis:")
        print("Top 10 States by Revenue:")
        print(state_summary.head(10))
        
        # Calculate concentration metrics
        total_revenue = state_summary['total_revenue'].sum()
        total_customers = state_summary['unique_customers'].sum()
        total_orders = state_summary['total_orders'].sum()
        
        # Top 3 states concentration
        top3_revenue_share = state_summary.head(3)['total_revenue'].sum() / total_revenue * 100
        top3_customer_share = state_summary.head(3)['unique_customers'].sum() / total_customers * 100
        top3_order_share = state_summary.head(3)['total_orders'].sum() / total_orders * 100
        
        print(f"\n🌍 Geographic Concentration (Top 3 States):")
        print(f"Revenue share: {top3_revenue_share:.1f}%")
        print(f"Customer share: {top3_customer_share:.1f}%")
        print(f"Order share: {top3_order_share:.1f}%")
        
        # Identify potential biases
        if 'SP' in state_summary.index:
            sp_dominance = state_summary.loc['SP', 'total_revenue'] / total_revenue * 100
            print(f"\n🏙️  SP (São Paulo) alone represents {sp_dominance:.1f}% of total revenue")
            print(f"This indicates significant geographic concentration that could bias analysis")
        else:
            print("\n⚠️  SP not found in state data - check data quality")
            
        return {
            'state_summary': state_summary,
            'customer_orders': customer_orders,
            'concentration_metrics': {
                'top3_revenue_share': top3_revenue_share,
                'top3_customer_share': top3_customer_share,
                'top3_order_share': top3_order_share
            }
        }
        
    except Exception as e:
        print(f"❌ Error in geography analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the fixed analysis
    results = fix_geography_analysis()
    
    if results:
        print("\n✅ Geography analysis completed successfully!")
        print("You can now continue with the rest of the notebook.")
    else:
        print("\n❌ Geography analysis failed. Check the error messages above.")