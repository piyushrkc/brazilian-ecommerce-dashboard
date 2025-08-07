#!/usr/bin/env python3
"""
Validation Script for Jupyter Notebook Variables
===============================================

This script checks that all key variables used in the final summary cell (cell 21)
are properly defined and available after running the notebook.

Run this after fixing the geography analysis to ensure everything works correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def validate_notebook_variables():
    """
    Validate that all required variables are properly defined
    """
    print("🔍 Validating Notebook Variables and Dependencies...")
    print("=" * 60)
    
    # List of critical variables that need to be defined
    required_variables = {
        'delivered_orders': 'DataFrame with delivered orders',
        'category_performance': 'DataFrame with category performance metrics',
        'rfm_data': 'DataFrame with RFM segmentation',
        'clv_by_segment': 'DataFrame with CLV analysis by segment',
        'top3_revenue_share': 'Top 3 states revenue share percentage',
        'top3_customer_share': 'Top 3 states customer share percentage',
        'top3_order_share': 'Top 3 states order share percentage',
        'sp_dominance': 'São Paulo revenue dominance percentage',
        'one_time_customers': 'Percentage of one-time customers',
        'total_clv': 'Total customer lifetime value'
    }
    
    # Variables that will be created
    created_vars = {}
    errors = []
    
    try:
        # Load datasets
        print("📊 Loading datasets...")
        orders = pd.read_csv('Data/olist_orders_dataset.csv')
        order_items = pd.read_csv('Data/olist_order_items_dataset.csv')
        customers = pd.read_csv('Data/olist_customers_dataset.csv')
        products = pd.read_csv('Data/olist_products_dataset.csv')
        payments = pd.read_csv('Data/olist_order_payments_dataset.csv')
        category_translation = pd.read_csv('Data/product_category_name_translation.csv')
        
        print("✅ All datasets loaded successfully")
        
        # Create delivered_orders
        print("\n📈 Creating delivered_orders...")
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
        orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
        orders['delivery_delay_days'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.days
        orders['is_late'] = orders['delivery_delay_days'] > 0
        
        delivered_orders = orders[orders['order_status'] == 'delivered'].copy()
        created_vars['delivered_orders'] = delivered_orders
        print(f"✅ delivered_orders created: {delivered_orders.shape}")
        
        # Create category_performance
        print("\n📊 Creating category_performance...")
        delivery_analysis = delivered_orders.merge(order_items, on='order_id')
        delivery_analysis = delivery_analysis.merge(products, on='product_id')
        delivery_analysis = delivery_analysis.merge(category_translation, on='product_category_name', how='left')
        
        category_performance = delivery_analysis.groupby('product_category_name_english').agg({
            'is_late': ['count', 'sum', 'mean'],
            'delivery_delay_days': ['mean', 'median'],
            'delivery_days': 'mean',
            'price': 'mean'
        }).round(2)
        
        category_performance.columns = ['total_orders', 'late_orders', 'late_rate', 'avg_delay_days', 
                                       'median_delay_days', 'avg_delivery_days', 'avg_price']
        category_performance = category_performance[category_performance['total_orders'] >= 100].sort_values('late_rate', ascending=False)
        created_vars['category_performance'] = category_performance
        print(f"✅ category_performance created: {category_performance.shape}")
        
        # Create customer_orders for RFM
        print("\n👥 Creating customer_orders for RFM...")
        customer_orders = orders.merge(order_items, on='order_id')
        payments_agg = payments.groupby('order_id')['payment_value'].sum().reset_index()
        customer_orders = customer_orders.merge(payments_agg, on='order_id', how='left')
        customer_orders = customer_orders.merge(customers, on='customer_id', how='left')
        
        print("✅ customer_orders created with customer info")
        
        # Create RFM data
        print("\n📊 Creating RFM data...")
        analysis_date = customer_orders['order_purchase_timestamp'].max()
        
        rfm_data = customer_orders.groupby('customer_unique_id').agg({
            'order_purchase_timestamp': ['max', 'count'],
            'payment_value': ['sum', 'mean']
        }).round(2)
        
        rfm_data.columns = ['last_purchase_date', 'frequency', 'total_spent', 'avg_order_value']
        rfm_data['recency'] = (analysis_date - rfm_data['last_purchase_date']).dt.days
        rfm_data['monetary'] = rfm_data['total_spent']
        
        # Create RFM scores
        rfm_data['R_score'] = pd.qcut(rfm_data['recency'], 5, labels=[5,4,3,2,1])
        rfm_data['F_score'] = pd.qcut(rfm_data['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm_data['M_score'] = pd.qcut(rfm_data['monetary'], 5, labels=[1,2,3,4,5])
        
        rfm_data['R_score'] = rfm_data['R_score'].astype(int)
        rfm_data['F_score'] = rfm_data['F_score'].astype(int)
        rfm_data['M_score'] = rfm_data['M_score'].astype(int)
        
        # Segment customers
        def segment_customers(row):
            if row['R_score'] >= 4 and row['F_score'] >= 4 and row['M_score'] >= 4:
                return 'Champions'
            elif row['R_score'] >= 3 and row['F_score'] >= 3 and row['M_score'] >= 3:
                return 'Loyal Customers'
            elif row['R_score'] >= 4 and row['F_score'] <= 2:
                return 'New Customers'
            elif row['R_score'] >= 3 and row['F_score'] >= 2 and row['M_score'] >= 2:
                return 'Potential Loyalists'
            elif row['R_score'] >= 3 and row['F_score'] <= 2:
                return 'Promising'
            elif row['R_score'] <= 2 and row['F_score'] >= 3:
                return 'At Risk'
            elif row['R_score'] <= 2 and row['F_score'] <= 2 and row['M_score'] >= 3:
                return "Can't Lose Them"
            elif row['R_score'] <= 2 and row['F_score'] <= 2:
                return 'Lost Customers'
            else:
                return 'Others'
        
        rfm_data['segment'] = rfm_data.apply(segment_customers, axis=1)
        created_vars['rfm_data'] = rfm_data
        print(f"✅ rfm_data created: {rfm_data.shape}")
        
        # Create CLV data
        print("\n💰 Creating CLV analysis...")
        customer_lifespan = customer_orders.groupby('customer_unique_id')['order_purchase_timestamp'].agg(['min', 'max'])
        customer_lifespan['lifespan_days'] = (customer_lifespan['max'] - customer_lifespan['min']).dt.days
        customer_lifespan['lifespan_days'] = customer_lifespan['lifespan_days'].fillna(0)
        
        rfm_clv = rfm_data.merge(customer_lifespan[['lifespan_days']], left_index=True, right_index=True)
        rfm_clv['purchase_frequency_per_day'] = rfm_clv['frequency'] / (rfm_clv['lifespan_days'] + 1)
        rfm_clv['estimated_annual_clv'] = rfm_clv['avg_order_value'] * rfm_clv['purchase_frequency_per_day'] * 365
        
        single_purchase_mask = rfm_clv['frequency'] == 1
        rfm_clv.loc[single_purchase_mask, 'estimated_annual_clv'] = rfm_clv.loc[single_purchase_mask, 'avg_order_value'] * 0.5
        
        clv_by_segment = rfm_clv.groupby('segment').agg({
            'estimated_annual_clv': ['mean', 'median', 'sum'],
            'frequency': 'mean',
            'avg_order_value': 'mean',
            'lifespan_days': 'mean'
        }).round(2)
        
        clv_by_segment.columns = ['avg_clv', 'median_clv', 'total_clv', 'avg_frequency', 'avg_order_value', 'avg_lifespan_days']
        clv_by_segment['customer_count'] = rfm_clv['segment'].value_counts()
        created_vars['clv_by_segment'] = clv_by_segment
        print(f"✅ clv_by_segment created: {clv_by_segment.shape}")
        
        # Calculate total CLV
        total_clv = clv_by_segment['total_clv'].sum()
        created_vars['total_clv'] = total_clv
        print(f"✅ total_clv calculated: R$ {total_clv:,.2f}")
        
        # Geographic analysis
        print("\n🌍 Creating geographic metrics...")
        if 'customer_state' not in customer_orders.columns:
            print("❌ customer_state missing - this is the issue we fixed!")
            errors.append("customer_state column missing in customer_orders")
        else:
            state_summary = customer_orders.groupby('customer_state').agg({
                'customer_id': 'nunique',
                'order_id': 'nunique',
                'payment_value': ['sum', 'mean']
            }).round(2)
            
            state_summary.columns = ['unique_customers', 'total_orders', 'total_revenue', 'avg_order_value']
            state_summary = state_summary.sort_values('total_revenue', ascending=False)
            
            total_revenue = state_summary['total_revenue'].sum()
            total_customers = state_summary['unique_customers'].sum()
            total_orders = state_summary['total_orders'].sum()
            
            top3_revenue_share = state_summary.head(3)['total_revenue'].sum() / total_revenue * 100
            top3_customer_share = state_summary.head(3)['unique_customers'].sum() / total_customers * 100
            top3_order_share = state_summary.head(3)['total_orders'].sum() / total_orders * 100
            
            created_vars['top3_revenue_share'] = top3_revenue_share
            created_vars['top3_customer_share'] = top3_customer_share
            created_vars['top3_order_share'] = top3_order_share
            
            if 'SP' in state_summary.index:
                sp_dominance = state_summary.loc['SP', 'total_revenue'] / total_revenue * 100
                created_vars['sp_dominance'] = sp_dominance
                print(f"✅ Geographic metrics calculated (SP: {sp_dominance:.1f}%)")
            else:
                errors.append("SP not found in state summary")
        
        # Customer retention
        print("\n🔄 Creating retention metrics...")
        customer_behavior = customer_orders.groupby('customer_unique_id').agg({
            'order_purchase_timestamp': ['min', 'max', 'count'],
            'payment_value': 'sum'
        })
        customer_behavior.columns = ['first_purchase', 'last_purchase', 'total_orders', 'total_spent']
        
        one_time_customers = (customer_behavior['total_orders'] == 1).sum() / len(customer_behavior) * 100
        created_vars['one_time_customers'] = one_time_customers
        print(f"✅ one_time_customers calculated: {one_time_customers:.1f}%")
        
    except Exception as e:
        errors.append(f"Error during validation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ Successfully created {len(created_vars)} variables:")
    for var_name, var_value in created_vars.items():
        if isinstance(var_value, pd.DataFrame):
            print(f"   • {var_name}: DataFrame with shape {var_value.shape}")
        elif isinstance(var_value, (int, float)):
            print(f"   • {var_name}: {var_value:.2f}")
        else:
            print(f"   • {var_name}: {type(var_value).__name__}")
    
    if errors:
        print(f"\n❌ Errors found ({len(errors)}):")
        for error in errors:
            print(f"   • {error}")
    else:
        print("\n🎉 All variables validated successfully!")
        print("✅ The final summary cell (cell 21) should work correctly!")
    
    # Check for missing required variables
    missing_vars = set(required_variables.keys()) - set(created_vars.keys())
    if missing_vars:
        print(f"\n⚠️  Missing variables: {', '.join(missing_vars)}")
        print("These need to be created for the final cell to work properly")
    
    return created_vars, errors

if __name__ == "__main__":
    print("Brazilian E-Commerce Notebook Validation")
    print("=" * 60)
    created_vars, errors = validate_notebook_variables()
    
    if not errors:
        print("\n✅ Validation complete - notebook should run successfully!")
    else:
        print("\n❌ Issues found - please fix before running the final cell")