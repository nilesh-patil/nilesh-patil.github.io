---
layout: post
title: "Visualizing distributions"
comments: true
modified:
categories: blog
excerpt: 'Common visualization examples for distributions'
tags: [data, distribution, visualization, seaborn]
image:
  feature:
date: 2017-01-14T15:39:55-04:00
modified: 2017-01-14T14:19:19-04:00
---

###### Sections:

* [1. Histogram](#histogram)
* [2. Scatter plot](#scatter-plot)
* [3. Density plot](#density-plot)
* [4. Boxplot](#boxplot)
* [5. Violin-plot](#violin-plot)
* [6. Heatmap](#heatmap)
* [7. Rugs](#rugs)

Once you have your data, usually you start by building summaries, checking for outliers, anomalies in the data & visualizing it from different angles. Here, we'll look at a few common approaches to visualize distributions (in a highly general sense).



```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('./indicators.txt',sep='\t')
countries = pd.read_csv('./country_classification_worldbank.txt',sep='\t')

data.columns=['countryName']+[colname for colname in data.columns[1:]]
columns = data.columns

countries['countryName'] = countries.countryName.astype(str)
data['countryName'] = data.countryName.astype(str)
data = data.merge(countries,how='left',on='countryName')
data = data[~data.Region.isnull()]
```

#### Histograms:


```python
sns.plt.figure(figsize=(8,5))
sns.set(style="whitegrid",palette="pastel", color_codes=True)
sns.set_style('whitegrid')

sns.plt.hist(data['Life expectancy at birth- years'],20);
sns.plt.title('Life expectancy at birth- years');
```


![png](\images\blog\distributions\output_5_0.png)


#### Scatter Plot:


```python
sns.plt.figure(figsize=(8,5))
sns.set(style="whitegrid", palette="pastel", color_codes=True)
sns.set_style('whitegrid')

sns.plt.scatter(x='MaleSuicide Rate 100k people',
                y='Female Suicide Rate 100k people',
                data=data,
                s=75,
                alpha=0.75);
sns.plt.title('Femal vs Male suicide rate(per 100k)');
sns.plt.xlabel('Male');
sns.plt.ylabel('Female');
```


![png](\images\blog\distributions\output_7_0.png)


#### Density plot:


```python
sns.plt.figure(figsize=(8,4))
sns.set(style="whitegrid", palette="pastel", color_codes=True)
sns.set_style('whitegrid')

sns.kdeplot(data['Life expectancy at birth- years'],legend=False);
sns.plt.title('Life expectancy at birth - years');
sns.plt.xlabel('Years');
sns.plt.ylabel('Fraction');
```


![png](\images\blog\distributions\output_9_0.png)


#### Boxplot:


```python
sns.plt.figure(figsize=(8,4))
sns.set(style="whitegrid", palette="pastel", color_codes=True)
sns.boxplot(x='Region',
            y='Life expectancy at birth- years',
            palette=sns.color_palette("Blues"),
            order=['Sub-Saharan Africa','South Asia',
                   'East Asia & Pacific','Latin America & Caribbean',
                   'Middle East & North Africa','Europe & Central Asia',
                   'North America'],
            width=0.4,
            fliersize=3,
            data=data);
sns.plt.title('Life expectancy (by region)')
sns.plt.ylabel('Life expectancy at birth (years)')
sns.plt.xticks(rotation=30);
```


![png](\images\blog\distributions\output_11_0.png)


#### Violin plot:


```python
sns.plt.figure(figsize=(8,5))
sns.set(style="whitegrid", palette="pastel", color_codes=True);
sns.violinplot(x='Income group',
               y='Secondary 2008-2014',
               width=0.6,
               inner='quart',
               color='b',
               order=['Low income','Lower middle income',
                     'Upper middle income','High income'],
               data=data);
sns.plt.title('Life Expectancy (by income group)');
sns.plt.ylabel('Life expectancy at birth (years)');
```


![png](\images\blog\distributions\output_13_0.png)


An interesting variation of violin plot that I use for comparing two distributions across multiple groups is by splitting them in two regions, one for each distribution. By default, `seaborn` supports this by taking in the `hue` parameter. The way we compare two different variables is by reshaping our dataset as shown below.


```python
variables = {'Primary-2008-2014':'b','Secondary 2008-2014':'y'}
data_plot = pd.melt(data,
                    value_vars=variables.keys(),
                    id_vars=['countryName','Income group'])

sns.plt.figure(figsize=(10,8))
sns.set(style="whitegrid", palette="pastel", color_codes=True)
sns.violinplot(x='Income group',
               y='value',
               hue='variable',
               split=True,
               inner="quart",
               order=['Low income','Lower middle income',
                     'Upper middle income','High income'],
               palette=variables,
               data=data_plot,);
sns.plt.title('Primary & Secondary enrollment (by income)');
sns.plt.ylabel('');
```


![png](\images\blog\distributions\output_15_0.png)



```python
variables = {'Primary-2008-2014':'b','Secondary 2008-2014':'y'}
data_plot = pd.melt(data,
                    value_vars=variables.keys(),
                    id_vars=['countryName','Region'])

sns.plt.figure(figsize=(10,8))
sns.set(style="whitegrid", palette="pastel", color_codes=True)
sns.violinplot(x='Region',
               y='value',
               hue='variable',
               split=True,
               inner="quart",
               palette=variables,
               data=data_plot,);
sns.plt.title('Primary & Secondary enrollment (by income)');
sns.plt.ylabel('');
sns.plt.xticks(rotation=30);
```


![png](\images\blog\distributions\output_16_0.png)


#### Heatmap:


```python
order=['Low income','Lower middle income',
                     'Upper middle income','High income']

data_plot = pd.pivot_table(data,
               values='Research and development expenditure  2005-2012',
               columns='Income group',
               index='Region')

figure = sns.plt.figure(figsize=(10,4))
sns.set(style="whitegrid", palette="pastel", color_codes=True)

figure.add_subplot(121)
sns.heatmap(data_plot[order],
            xticklabels=['LI','LMI','UMI','HI'],);
sns.plt.ylabel('');
sns.plt.xlabel('');
sns.plt.title('Expenditure of R&D (%GDP)');


data_plot = pd.pivot_table(data,
               values='Public expenditure on education Percentange GDP',
               columns='Income group',
               index='Region')

figure.add_subplot(122)
sns.heatmap(data_plot[order],
            xticklabels=['LI','LMI','UMI','HI'],
            yticklabels=False);
sns.plt.ylabel('');
sns.plt.xlabel('');
sns.plt.title('Expenditure on education (%GDP)');
```


![png](\images\blog\distributions\output_18_0.png)


#### Rugs:
