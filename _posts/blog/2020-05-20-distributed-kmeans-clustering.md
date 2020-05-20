---
layout: post
title: "Distributed K-Means Clustering in Python"
modified:
categories: blog
excerpt: 'Implementing scalable k-means clustering using distributed computing frameworks'
tags: [machine-learning, clustering, distributed-computing, python, pyspark, dask]
image:
  feature:
date: 2020-05-20T10:00:00-00:00
modified: 2020-05-20T10:00:00-00:00
---

## Index

[Introduction](#introduction)
[K-Means Algorithm Overview](#k-means-algorithm-overview)
[Challenges with Large-Scale Data](#challenges-with-large-scale-data)
[Distributed K-Means with PySpark](#distributed-k-means-with-pyspark)
[Distributed K-Means with Dask](#distributed-k-means-with-dask)
[Performance Comparison](#performance-comparison)
[Best Practices](#best-practices)
[Conclusion](#conclusion)

## Introduction

K-means clustering is one of the most widely used unsupervised machine learning algorithms for partitioning data into k clusters. While the algorithm is conceptually simple and computationally efficient for moderate-sized datasets, it faces significant challenges when dealing with big data scenarios where datasets can contain millions or billions of data points.

In this post, we'll explore how to implement distributed k-means clustering in Python using popular frameworks like **PySpark** and **Dask**, enabling us to handle massive datasets that don't fit into memory on a single machine.

## K-Means Algorithm Overview

Before diving into distributed implementations, let's quickly review the standard k-means algorithm:

1. **Initialize** k cluster centroids randomly
2. **Assignment Step**: Assign each data point to the nearest centroid
3. **Update Step**: Recalculate centroids as the mean of assigned points
4. **Repeat** steps 2-3 until convergence or maximum iterations reached

```python
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Standard scikit-learn k-means for small datasets
def standard_kmeans_example():
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    
    # Apply k-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], 
                kmeans.cluster_centers_[:, 1], 
                c='red', marker='x', s=200)
    plt.title('Standard K-Means Clustering')
    plt.show()
    
    return kmeans, labels
```

## Challenges with Large-Scale Data

When dealing with big data, traditional k-means implementations face several challenges:

1. **Memory Constraints**: Large datasets may not fit into memory
2. **Computational Complexity**: O(n*k*d*i) time complexity becomes prohibitive
3. **I/O Bottlenecks**: Reading massive datasets from disk
4. **Scalability**: Single-machine limitations

## Distributed K-Means with PySpark

Apache Spark provides excellent support for distributed k-means clustering through its MLlib library. Here's how to implement it:

### Setting Up PySpark Environment

```python
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
import pyspark.sql.functions as F

# Initialize Spark session
def create_spark_session():
    spark = SparkSession.builder \
        .appName("DistributedKMeans") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark
```

### Implementing Distributed K-Means

```python
def distributed_kmeans_pyspark(spark, data_path, k=3, max_iter=100):
    """
    Perform distributed k-means clustering using PySpark
    
    Parameters:
    -----------
    spark : SparkSession
        Active Spark session
    data_path : str
        Path to the dataset (CSV, Parquet, etc.)
    k : int
        Number of clusters
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    model : KMeansModel
        Trained k-means model
    predictions : DataFrame
        DataFrame with cluster assignments
    """
    
    # Load data
    df = spark.read.option("header", "true").csv(data_path)
    
    # Convert string columns to numeric (if needed)
    numeric_cols = [col_name for col_name in df.columns 
                    if col_name not in ['id', 'label']]
    
    for col_name in numeric_cols:
        df = df.withColumn(col_name, col(col_name).cast("double"))
    
    # Create feature vector
    assembler = VectorAssembler(
        inputCols=numeric_cols,
        outputCol="features"
    )
    df_vectorized = assembler.transform(df)
    
    # Initialize and train k-means model
    kmeans = SparkKMeans(
        k=k,
        maxIter=max_iter,
        seed=42,
        featuresCol="features",
        predictionCol="cluster"
    )
    
    model = kmeans.fit(df_vectorized)
    
    # Make predictions
    predictions = model.transform(df_vectorized)
    
    # Display cluster centers
    centers = model.clusterCenters()
    print(f"Cluster Centers:")
    for i, center in enumerate(centers):
        print(f"Cluster {i}: {center}")
    
    # Calculate Within Set Sum of Squared Errors (WSSSE)
    wssse = model.summary.trainingCost
    print(f"Within Set Sum of Squared Errors: {wssse}")
    
    return model, predictions

# Example usage
def run_pyspark_example():
    spark = create_spark_session()
    
    # Generate sample distributed dataset
    sample_data = spark.range(0, 100000).select(
        F.rand(seed=42).alias("feature1"),
        F.rand(seed=43).alias("feature2"),
        F.rand(seed=44).alias("feature3")
    )
    
    # Save to temporary location for demonstration
    sample_data.write.mode("overwrite").option("header", "true").csv("/tmp/sample_data")
    
    # Run distributed k-means
    model, predictions = distributed_kmeans_pyspark(
        spark, "/tmp/sample_data", k=5, max_iter=50
    )
    
    # Show sample predictions
    predictions.select("feature1", "feature2", "feature3", "cluster").show(20)
    
    spark.stop()
    return model, predictions
```

### Advanced PySpark K-Means with Custom Initialization

```python
def advanced_kmeans_pyspark(spark, df, k=3, init_method="k-means++"):
    """
    Advanced k-means implementation with custom initialization strategies
    """
    
    if init_method == "k-means++":
        # Use PySpark's default k-means++ initialization
        kmeans = SparkKMeans(k=k, initMode="k-means||", initSteps=2)
    elif init_method == "random":
        kmeans = SparkKMeans(k=k, initMode="random")
    
    # Add convergence tolerance
    kmeans.setTol(1e-4)
    
    # Train model
    model = kmeans.fit(df)
    
    # Evaluate model performance
    silhouette_evaluator = ClusteringEvaluator()
    predictions = model.transform(df)
    silhouette = silhouette_evaluator.evaluate(predictions)
    
    print(f"Silhouette Score: {silhouette}")
    
    return model, predictions
```

## Distributed K-Means with Dask

Dask provides another excellent framework for distributed computing in Python, with a more Pythonic API:

### Setting Up Dask Environment

```python
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, as_completed
from dask_ml.cluster import KMeans as DaskKMeans
import numpy as np

def create_dask_client(n_workers=4):
    """
    Create a Dask client for distributed computing
    """
    client = Client(n_workers=n_workers, threads_per_worker=2)
    print(f"Dask dashboard available at: {client.dashboard_link}")
    return client
```

### Implementing Distributed K-Means with Dask

```python
def distributed_kmeans_dask(data_path, k=3, max_iter=100, chunk_size="100MB"):
    """
    Perform distributed k-means clustering using Dask
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset
    k : int
        Number of clusters
    max_iter : int
        Maximum number of iterations
    chunk_size : str
        Size of data chunks for processing
        
    Returns:
    --------
    model : DaskKMeans
        Trained k-means model
    labels : dask.array
        Cluster assignments
    """
    
    # Load data with Dask
    df = dd.read_csv(data_path)
    
    # Convert to numpy array for clustering
    # Exclude non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].to_dask_array(lengths=True)
    
    # Initialize Dask k-means
    kmeans = DaskKMeans(
        n_clusters=k,
        max_iter=max_iter,
        random_state=42,
        init_max_iter=3  # For k-means++ initialization
    )
    
    # Fit the model
    print("Training distributed k-means model...")
    kmeans.fit(X)
    
    # Predict cluster labels
    labels = kmeans.predict(X)
    
    # Get cluster centers
    centers = kmeans.cluster_centers_
    print(f"Cluster centers shape: {centers.shape}")
    
    # Calculate inertia (within-cluster sum of squares)
    inertia = kmeans.inertia_
    print(f"Inertia: {inertia}")
    
    return kmeans, labels

# Example with synthetic data generation
def generate_large_dataset_dask(n_samples=1000000, n_features=10, n_centers=5):
    """
    Generate a large synthetic dataset using Dask
    """
    from sklearn.datasets import make_blobs
    
    # Generate data in chunks
    chunk_size = 100000
    chunks = []
    
    for i in range(0, n_samples, chunk_size):
        current_chunk_size = min(chunk_size, n_samples - i)
        X_chunk, _ = make_blobs(
            n_samples=current_chunk_size,
            centers=n_centers,
            n_features=n_features,
            random_state=42 + i,
            cluster_std=1.5
        )
        chunks.append(da.from_array(X_chunk, chunks=(current_chunk_size, n_features)))
    
    # Concatenate chunks
    X_large = da.concatenate(chunks, axis=0)
    return X_large

def run_dask_example():
    """
    Complete example of distributed k-means with Dask
    """
    # Create Dask client
    client = create_dask_client(n_workers=4)
    
    try:
        # Generate large synthetic dataset
        print("Generating large synthetic dataset...")
        X = generate_large_dataset_dask(n_samples=500000, n_features=8, n_centers=4)
        
        # Apply k-means clustering
        print("Applying distributed k-means...")
        kmeans = DaskKMeans(n_clusters=4, random_state=42)
        
        # Fit and predict
        labels = kmeans.fit_predict(X)
        
        # Compute results
        unique_labels = da.unique(labels).compute()
        print(f"Unique cluster labels: {unique_labels}")
        
        # Calculate cluster statistics
        centers = kmeans.cluster_centers_
        print(f"Cluster centers:\n{centers}")
        
        return kmeans, labels
        
    finally:
        # Clean up
        client.close()
```

### Incremental K-Means with Dask

```python
def incremental_kmeans_dask(data_stream, k=3, batch_size=10000):
    """
    Implement incremental k-means for streaming data
    """
    from dask_ml.cluster import KMeans
    
    # Initialize model
    kmeans = KMeans(n_clusters=k, init_max_iter=1)
    
    # Process data in batches
    for batch in data_stream:
        # Partial fit on current batch
        kmeans.partial_fit(batch)
        
        # Optional: Track convergence
        if hasattr(kmeans, 'inertia_'):
            print(f"Current inertia: {kmeans.inertia_}")
    
    return kmeans
```

## Performance Comparison

Let's compare the performance of different implementations:

```python
import time
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def performance_comparison():
    """
    Compare performance of different k-means implementations
    """
    # Generate test data
    sizes = [10000, 50000, 100000]
    results = {}
    
    for size in sizes:
        print(f"\nTesting with {size} samples...")
        X, _ = make_blobs(n_samples=size, centers=5, n_features=10, random_state=42)
        
        # Scikit-learn (single-threaded)
        start_time = time.time()
        sklearn_kmeans = KMeans(n_clusters=5, random_state=42)
        sklearn_kmeans.fit(X)
        sklearn_time = time.time() - start_time
        
        # Dask (if data fits in memory)
        start_time = time.time()
        X_dask = da.from_array(X, chunks=(10000, 10))
        dask_kmeans = DaskKMeans(n_clusters=5, random_state=42)
        dask_kmeans.fit(X_dask)
        dask_time = time.time() - start_time
        
        results[size] = {
            'sklearn': sklearn_time,
            'dask': dask_time,
            'speedup': sklearn_time / dask_time
        }
        
        print(f"Scikit-learn: {sklearn_time:.2f}s")
        print(f"Dask: {dask_time:.2f}s")
        print(f"Speedup: {sklearn_time/dask_time:.2f}x")
    
    return results
```

## Best Practices

### 1. Data Preprocessing
```python
def preprocess_for_clustering(df):
    """
    Best practices for data preprocessing
    """
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Remove outliers (optional)
    from scipy import stats
    z_scores = np.abs(stats.zscore(df_scaled))
    df_clean = df_scaled[(z_scores < 3).all(axis=1)]
    
    return df_clean
```

### 2. Optimal Number of Clusters
```python
def find_optimal_k_distributed(X, max_k=10):
    """
    Find optimal number of clusters using elbow method
    """
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = DaskKMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()
    
    return k_range, inertias
```

### 3. Memory Management
```python
def optimize_memory_usage():
    """
    Tips for optimizing memory usage in distributed k-means
    """
    
    # 1. Use appropriate chunk sizes
    chunk_size = "100MB"  # Adjust based on available memory
    
    # 2. Use float32 instead of float64 when possible
    dtype = np.float32
    
    # 3. Persist intermediate results strategically
    # df.persist()  # Only when data will be reused multiple times
    
    # 4. Use garbage collection
    import gc
    gc.collect()
    
    return {
        'chunk_size': chunk_size,
        'dtype': dtype
    }
```

## Conclusion

Distributed k-means clustering is essential for handling large-scale datasets that exceed single-machine capabilities. Both PySpark and Dask offer robust solutions:

**PySpark MLlib** is ideal when:
- Working with very large datasets (>1TB)
- Integration with existing Spark ecosystem
- Need for production-grade fault tolerance

**Dask** is preferred when:
- Working with Python-centric workflows
- Need for interactive development
- Integration with existing NumPy/Pandas code

**Key Takeaways:**

1. **Preprocessing** is crucial for distributed clustering success
2. **Chunk size** optimization significantly impacts performance
3. **Initialization methods** (k-means++) are important for convergence
4. **Monitoring** convergence and performance metrics is essential
5. **Memory management** becomes critical at scale

The choice between frameworks depends on your specific use case, data size, and existing infrastructure. Both approaches can handle datasets that would be impossible to process on a single machine, making k-means clustering accessible for big data applications.

```python
# Final example: Complete pipeline
def complete_distributed_kmeans_pipeline(data_path, framework='dask'):
    """
    Complete pipeline for distributed k-means clustering
    """
    if framework == 'dask':
        client = create_dask_client()
        try:
            # Load and preprocess data
            df = dd.read_csv(data_path)
            X = preprocess_for_clustering(df.values)
            
            # Find optimal k
            k_range, inertias = find_optimal_k_distributed(X)
            optimal_k = find_elbow_point(k_range, inertias)
            
            # Train final model
            kmeans = DaskKMeans(n_clusters=optimal_k, random_state=42)
            labels = kmeans.fit_predict(X)
            
            return kmeans, labels
        finally:
            client.close()
    
    elif framework == 'pyspark':
        spark = create_spark_session()
        try:
            model, predictions = distributed_kmeans_pyspark(
                spark, data_path, k=optimal_k
            )
            return model, predictions
        finally:
            spark.stop()
```

This comprehensive approach to distributed k-means clustering will help you tackle large-scale clustering problems efficiently and effectively. 