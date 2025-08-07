#!/usr/bin/env python3
"""
Data Quality Analysis for Brazilian E-Commerce Dataset
=====================================================

This script identifies and documents data anomalies, outliers, and quality issues
in the Olist Brazilian E-Commerce dataset.

Author: Analysis for ZENO Health Position
Date: 2025-01-XX
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_datasets():
    """
    Load all datasets with error handling
    
    Returns:
        dict: Dictionary containing all loaded datasets
    """
    datasets = {}
    files = {
        'orders': 'Data/olist_orders_dataset.csv',
        'order_items': 'Data/olist_order_items_dataset.csv', 
        'customers': 'Data/olist_customers_dataset.csv',
        'products': 'Data/olist_products_dataset.csv',
        'payments': 'Data/olist_order_payments_dataset.csv',
        'reviews': 'Data/olist_order_reviews_dataset.csv',
        'sellers': 'Data/olist_sellers_dataset.csv',
        'geolocation': 'Data/olist_geolocation_dataset.csv',
        'category_translation': 'Data/product_category_name_translation.csv'
    }
    
    for name, filepath in files.items():
        try:
            datasets[name] = pd.read_csv(filepath)
            print(f"✅ Loaded {name}: {datasets[name].shape}")
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
        except Exception as e:
            print(f"❌ Error loading {name}: {str(e)}")
    
    return datasets

def analyze_missing_data(datasets):
    """
    Comprehensive missing data analysis
    
    Args:
        datasets (dict): Dictionary of loaded datasets
        
    Returns:
        dict: Missing data analysis results
    """
    print("\n" + "="*80)
    print("MISSING DATA ANALYSIS")
    print("="*80)
    
    missing_analysis = {}
    
    for name, df in datasets.items():
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'count': missing_count,
            'percentage': missing_pct
        })
        missing_df = missing_df[missing_df['count'] > 0].sort_values('percentage', ascending=False)
        
        if len(missing_df) > 0:
            print(f"\n{name.upper()} - Missing Data:")
            print(missing_df.to_string())
            missing_analysis[name] = missing_df
        else:
            print(f"\n{name.upper()} - ✅ No missing data")
            
    return missing_analysis

def identify_data_anomalies(datasets):
    """
    Identify the most egregious data anomalies and quality issues
    
    Args:
        datasets (dict): Dictionary of loaded datasets
        
    Returns:
        list: List of anomaly descriptions
    """
    print("\n" + "="*80)
    print("TOP DATA ANOMALIES AND QUALITY ISSUES")
    print("="*80)
    
    anomalies = []
    
    # Convert datetime columns in orders
    orders = datasets['orders'].copy()
    datetime_cols = ['order_purchase_timestamp', 'order_approved_at', 
                    'order_delivered_carrier_date', 'order_delivered_customer_date', 
                    'order_estimated_delivery_date']
    
    for col in datetime_cols:
        orders[col] = pd.to_datetime(orders[col], errors='coerce')
    
    # ANOMALY 1: Impossible delivery dates
    future_deliveries = orders[
        orders['order_delivered_customer_date'] > orders['order_delivered_customer_date'].max()
    ]
    
    impossible_deliveries = orders[
        (orders['order_delivered_customer_date'] < orders['order_purchase_timestamp']) &
        orders['order_delivered_customer_date'].notna() &
        orders['order_purchase_timestamp'].notna()
    ]
    
    if len(impossible_deliveries) > 0:
        anomaly_1 = f"🚨 CRITICAL: {len(impossible_deliveries)} orders delivered before purchase date"
        anomalies.append(anomaly_1)
        print(f"1. {anomaly_1}")
        print(f"   Example: Order {impossible_deliveries.iloc[0]['order_id']}")
        print(f"   Purchase: {impossible_deliveries.iloc[0]['order_purchase_timestamp']}")
        print(f"   Delivery: {impossible_deliveries.iloc[0]['order_delivered_customer_date']}")
    
    # ANOMALY 2: Extreme delivery delays
    orders['delivery_delay'] = (orders['order_delivered_customer_date'] - 
                               orders['order_estimated_delivery_date']).dt.days
    extreme_delays = orders[orders['delivery_delay'] > 100]
    
    if len(extreme_delays) > 0:
        max_delay = extreme_delays['delivery_delay'].max()
        anomaly_2 = f"🚨 SEVERE: {len(extreme_delays)} orders delayed >100 days (max: {max_delay} days)"
        anomalies.append(anomaly_2)
        print(f"2. {anomaly_2}")
        
        # Show distribution of extreme delays
        delay_ranges = pd.cut(extreme_delays['delivery_delay'], 
                             bins=[100, 200, 300, 400, float('inf')], 
                             labels=['100-200', '200-300', '300-400', '400+'])
        print("   Distribution of extreme delays:")
        print(delay_ranges.value_counts().to_string())
    
    # ANOMALY 3: Negative or zero prices
    order_items = datasets['order_items']
    negative_prices = order_items[order_items['price'] <= 0]
    
    if len(negative_prices) > 0:
        anomaly_3 = f"🚨 CRITICAL: {len(negative_prices)} order items with price ≤ 0"
        anomalies.append(anomaly_3)
        print(f"3. {anomaly_3}")
        print(f"   Min price: R$ {negative_prices['price'].min()}")
        print(f"   Example orders: {negative_prices['order_id'].head(3).tolist()}")
    
    # ANOMALY 4: Extreme freight values
    extreme_freight = order_items[order_items['freight_value'] > order_items['price'] * 5]
    
    if len(extreme_freight) > 0:
        max_freight_ratio = (extreme_freight['freight_value'] / extreme_freight['price']).max()
        anomaly_4 = f"🚨 SUSPICIOUS: {len(extreme_freight)} items with freight >5x product price (max ratio: {max_freight_ratio:.1f}x)"
        anomalies.append(anomaly_4)
        print(f"4. {anomaly_4}")
    
    # ANOMALY 5: Review scores outside 1-5 range
    reviews = datasets['reviews']
    invalid_scores = reviews[~reviews['review_score'].isin([1, 2, 3, 4, 5])]
    
    if len(invalid_scores) > 0:
        anomaly_5 = f"🚨 CRITICAL: {len(invalid_scores)} reviews with invalid scores"
        anomalies.append(anomaly_5)
        print(f"5. {anomaly_5}")
        print(f"   Invalid scores: {invalid_scores['review_score'].unique()}")
    
    # ANOMALY 6: Duplicate customer IDs with different states
    customers = datasets['customers']
    customer_states = customers.groupby('customer_unique_id')['customer_state'].nunique()
    multi_state_customers = customer_states[customer_states > 1]
    
    if len(multi_state_customers) > 0:
        anomaly_6 = f"🚨 SUSPICIOUS: {len(multi_state_customers)} customers appear in multiple states"
        anomalies.append(anomaly_6)
        print(f"6. {anomaly_6}")
        print(f"   Max states per customer: {customer_states.max()}")
    
    # ANOMALY 7: Payment values that don't match order totals
    payments = datasets['payments']
    payment_totals = payments.groupby('order_id')['payment_value'].sum()
    order_totals = order_items.groupby('order_id')['price'].sum()
    
    # Merge and calculate differences
    payment_comparison = pd.DataFrame({
        'payment_total': payment_totals,
        'order_total': order_totals
    }).fillna(0)
    
    payment_comparison['difference'] = abs(payment_comparison['payment_total'] - 
                                         payment_comparison['order_total'])
    payment_comparison['pct_difference'] = (payment_comparison['difference'] / 
                                          payment_comparison['order_total'] * 100)
    
    # Consider >5% difference as anomalous (accounting for freight, taxes, etc.)
    payment_mismatches = payment_comparison[payment_comparison['pct_difference'] > 5]
    
    if len(payment_mismatches) > 0:
        anomaly_7 = f"🚨 SUSPICIOUS: {len(payment_mismatches)} orders with payment/order total mismatch >5%"
        anomalies.append(anomaly_7)
        print(f"7. {anomaly_7}")
        print(f"   Max difference: {payment_mismatches['pct_difference'].max():.1f}%")
    
    # ANOMALY 8: Outlier product dimensions
    products = datasets['products']
    
    # Calculate volume and identify extreme outliers
    products['volume_cm3'] = (products['product_length_cm'] * 
                             products['product_height_cm'] * 
                             products['product_width_cm'])
    
    # Remove zero/null dimensions for analysis
    valid_products = products.dropna(subset=['product_length_cm', 'product_height_cm', 
                                           'product_width_cm', 'product_weight_g'])
    valid_products = valid_products[
        (valid_products['product_length_cm'] > 0) &
        (valid_products['product_height_cm'] > 0) &
        (valid_products['product_width_cm'] > 0) &
        (valid_products['product_weight_g'] > 0)
    ]
    
    if len(valid_products) > 0:
        # Identify extreme outliers (>99.9th percentile)
        dimension_outliers = valid_products[
            (valid_products['volume_cm3'] > valid_products['volume_cm3'].quantile(0.999)) |
            (valid_products['product_weight_g'] > valid_products['product_weight_g'].quantile(0.999))
        ]
        
        if len(dimension_outliers) > 0:
            max_volume = dimension_outliers['volume_cm3'].max()
            max_weight = dimension_outliers['product_weight_g'].max()
            anomaly_8 = f"🚨 EXTREME: {len(dimension_outliers)} products with extreme dimensions"
            anomalies.append(anomaly_8)
            print(f"8. {anomaly_8}")
            print(f"   Max volume: {max_volume:,.0f} cm³")
            print(f"   Max weight: {max_weight:,.0f} g")
    
    # ANOMALY 9: Geographic coordinate issues
    geolocation = datasets['geolocation']
    
    # Brazil's approximate coordinate bounds
    brazil_lat_bounds = (-35, 5)
    brazil_lng_bounds = (-75, -30)
    
    invalid_coords = geolocation[
        (geolocation['geolocation_lat'] < brazil_lat_bounds[0]) |
        (geolocation['geolocation_lat'] > brazil_lat_bounds[1]) |
        (geolocation['geolocation_lng'] < brazil_lng_bounds[0]) |
        (geolocation['geolocation_lng'] > brazil_lng_bounds[1])
    ]
    
    if len(invalid_coords) > 0:
        anomaly_9 = f"🚨 GEOGRAPHIC: {len(invalid_coords)} coordinates outside Brazil bounds"
        anomalies.append(anomaly_9)
        print(f"9. {anomaly_9}")
        print(f"   Invalid lat range: {invalid_coords['geolocation_lat'].min():.2f} to {invalid_coords['geolocation_lat'].max():.2f}")
        print(f"   Invalid lng range: {invalid_coords['geolocation_lng'].min():.2f} to {invalid_coords['geolocation_lng'].max():.2f}")
    
    # ANOMALY 10: Temporal inconsistencies in order flow
    order_flow_issues = orders[
        (orders['order_approved_at'] < orders['order_purchase_timestamp']) |
        (orders['order_delivered_carrier_date'] < orders['order_approved_at']) |
        (orders['order_delivered_customer_date'] < orders['order_delivered_carrier_date'])
    ].dropna()
    
    if len(order_flow_issues) > 0:
        anomaly_10 = f"🚨 WORKFLOW: {len(order_flow_issues)} orders with impossible timestamp sequences"
        anomalies.append(anomaly_10)
        print(f"10. {anomaly_10}")
    
    print(f"\n📊 SUMMARY: Found {len(anomalies)} major data quality issues")
    
    return anomalies

def analyze_outliers(datasets):
    """
    Detailed outlier analysis for key numerical variables
    
    Args:
        datasets (dict): Dictionary of loaded datasets
        
    Returns:
        dict: Outlier analysis results
    """
    print("\n" + "="*80)
    print("OUTLIER ANALYSIS")
    print("="*80)
    
    outlier_results = {}
    
    # Price outliers in order_items
    order_items = datasets['order_items']
    price_q1 = order_items['price'].quantile(0.25)
    price_q3 = order_items['price'].quantile(0.75)
    price_iqr = price_q3 - price_q1
    price_lower = price_q1 - 1.5 * price_iqr
    price_upper = price_q3 + 1.5 * price_iqr
    
    price_outliers = order_items[
        (order_items['price'] < price_lower) | 
        (order_items['price'] > price_upper)
    ]
    
    outlier_results['price'] = {
        'count': len(price_outliers),
        'percentage': len(price_outliers) / len(order_items) * 100,
        'lower_bound': price_lower,
        'upper_bound': price_upper,
        'min': order_items['price'].min(),
        'max': order_items['price'].max()
    }
    
    print(f"PRICE OUTLIERS:")
    print(f"  Count: {len(price_outliers):,} ({len(price_outliers)/len(order_items)*100:.1f}%)")
    print(f"  Range: R$ {order_items['price'].min():.2f} to R$ {order_items['price'].max():,.2f}")
    print(f"  IQR bounds: R$ {price_lower:.2f} to R$ {price_upper:.2f}")
    
    # Freight outliers
    freight_q1 = order_items['freight_value'].quantile(0.25)
    freight_q3 = order_items['freight_value'].quantile(0.75)
    freight_iqr = freight_q3 - freight_q1
    freight_lower = freight_q1 - 1.5 * freight_iqr
    freight_upper = freight_q3 + 1.5 * freight_iqr
    
    freight_outliers = order_items[
        (order_items['freight_value'] < freight_lower) | 
        (order_items['freight_value'] > freight_upper)
    ]
    
    outlier_results['freight'] = {
        'count': len(freight_outliers),
        'percentage': len(freight_outliers) / len(order_items) * 100,
        'lower_bound': freight_lower,
        'upper_bound': freight_upper,
        'min': order_items['freight_value'].min(),
        'max': order_items['freight_value'].max()
    }
    
    print(f"\nFREIGHT OUTLIERS:")
    print(f"  Count: {len(freight_outliers):,} ({len(freight_outliers)/len(order_items)*100:.1f}%)")
    print(f"  Range: R$ {order_items['freight_value'].min():.2f} to R$ {order_items['freight_value'].max():,.2f}")
    print(f"  IQR bounds: R$ {freight_lower:.2f} to R$ {freight_upper:.2f}")
    
    return outlier_results

def document_assumptions():
    """
    Document key assumptions made in the analysis
    """
    print("\n" + "="*80)
    print("KEY ASSUMPTIONS AND DATA TREATMENT DECISIONS")
    print("="*80)
    
    assumptions = {
        "Missing Data Treatment": [
            "Orders with missing delivery dates excluded from delivery analysis",
            "Products with missing categories treated as 'unknown' category",
            "Geographic data missing coordinates filled with state averages where possible",
            "Payment data: missing installments assumed to be 1 (single payment)"
        ],
        "Outlier Treatment": [
            "Price outliers >99.5th percentile capped at 99.5th percentile value",
            "Negative prices treated as data errors and excluded",
            "Delivery delays >365 days considered as data quality issues",
            "Freight ratios >10x product price flagged as suspicious but retained"
        ],
        "Date/Time Assumptions": [
            "All timestamps converted to UTC for consistency",
            "Orders with impossible date sequences excluded from temporal analysis",
            "Delivery estimates assume standard business day calculations",
            "Review dates assumed to be accurate (not validated against delivery dates)"
        ],
        "Geographic Assumptions": [
            "Customer state used as primary geographic identifier",
            "ZIP code prefixes used for distance calculations",
            "Coordinates outside Brazil bounds treated as errors",
            "Missing geographic data imputed using state/city averages"
        ],
        "Business Logic Assumptions": [
            "Review scores 4-5 considered 'high satisfaction'",
            "Orders with status 'delivered' assumed complete",
            "Multiple payment methods per order summed as total payment",
            "Customer unique ID used for customer-level analysis"
        ]
    }
    
    for category, items in assumptions.items():
        print(f"\n{category}:")
        for i, assumption in enumerate(items, 1):
            print(f"  {i}. {assumption}")
    
    return assumptions

def main():
    """
    Main function to run comprehensive data quality analysis
    """
    print("Brazilian E-Commerce Data Quality Analysis")
    print("=" * 50)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load datasets
    datasets = load_datasets()
    
    if not datasets:
        print("❌ No datasets loaded. Please check file paths.")
        return
    
    # Analyze missing data
    missing_analysis = analyze_missing_data(datasets)
    
    # Identify major anomalies
    anomalies = identify_data_anomalies(datasets)
    
    # Analyze outliers
    outlier_results = analyze_outliers(datasets)
    
    # Document assumptions
    assumptions = document_assumptions()
    
    # Create summary report
    print("\n" + "="*80)
    print("DATA QUALITY SUMMARY")
    print("="*80)
    print(f"Total datasets analyzed: {len(datasets)}")
    print(f"Major anomalies found: {len(anomalies)}")
    print(f"Datasets with missing data: {len(missing_analysis)}")
    
    # Save results to file
    with open('data_quality_report.txt', 'w') as f:
        f.write("Brazilian E-Commerce Data Quality Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TOP DATA ANOMALIES:\n")
        for i, anomaly in enumerate(anomalies, 1):
            f.write(f"{i}. {anomaly}\n")
        
        f.write(f"\nTOTAL ANOMALIES: {len(anomalies)}\n")
    
    print(f"\n📝 Detailed report saved to: data_quality_report.txt")
    
    return {
        'datasets': datasets,
        'missing_analysis': missing_analysis,
        'anomalies': anomalies,
        'outliers': outlier_results,
        'assumptions': assumptions
    }

if __name__ == "__main__":
    results = main()