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


### Connect to data:

```python
%pylab inline

import pandas as pd
import seaborn as sns
import sqlite3


db_path = './data/world-development-indicators/database.sqlite'

conn = sqlite3.connect(db_path)
db = conn.cursor()
db.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(db.fetchall())

data_countries = pd.read_sql_query('select * from Country',conn)
data_series = pd.read_sql_query('select * from Series',conn)
data_indicators = pd.read_sql_query('select * from Indicators',conn)

```

### Histogram:

##### Data Prep
```python
selected_indicators = ['Life expectancy at birth, female (years)',
                       'Life expectancy at birth, male (years)',
                       'Life expectancy at birth, total (years)']
countries = data_countries.CountryCode[data_countries.Region!=''].unique()
condition = data_indicators.IndicatorName.isin(selected_indicators)

data_plot = data_indicators.loc[condition,:]
condition = data_plot.CountryCode.isin(countries)
data_plot = data_plot.loc[condition,:]
data_plot.sort_values(['CountryName','IndicatorName','Year'], inplace=True)

data_plot = data_plot.groupby(['CountryName','IndicatorName'], as_index=False).last()
data_plot[['feature','type']] = data_plot['IndicatorName'].str.split(', ',expand=True)
data_plot.reset_index(inplace=True, drop=True)
```


##### Plot
```python
nbins = 15
sns.set(style="white",
        palette="pastel",
        color_codes=True,
        rc={'figure.figsize':(12,8),
            'figure.dpi':500})

sns.distplot(data_plot.Value[data_plot.type=='female (years)'], bins=nbins)
sns.distplot(data_plot.Value[data_plot.type=='male (years)'], bins=nbins)
sns.distplot(data_plot.Value[data_plot.type=='total (years)'], bins=nbins)
plt.legend(['Female', 'Male', 'Total'], bbox_to_anchor=(1.12,1.04))
plt.xlim((25,100))
plt.grid(color='black',linestyle='-.',linewidth=0.25)
plt.title('Life expectancy at birth ( In years )')
```


![Histogram](\images\blog\distributions\01.histogram.png){: .center-image height="300px" width="850px"}


### Scatter Plot:

##### Data Prep
```python

selected_indicators = ['Unemployment, female (% of female labor force)',
                       'Unemployment, male (% of male labor force)']

countries = data_countries.CountryCode[data_countries.Region!=''].unique()
condition = data_indicators.IndicatorName.isin(selected_indicators)

data_plot = data_indicators.loc[condition,:]
condition = data_plot.CountryCode.isin(countries)
data_plot = data_plot.loc[condition,:]
data_plot.sort_values(['CountryName','IndicatorName','Year'], inplace=True)

data_plot = data_plot.groupby(['CountryName','IndicatorName'], as_index=False).last()
data_plot[['feature','type']] = data_plot['IndicatorName'].str.split(', ',expand=True)
data_plot.reset_index(inplace=True, drop=True)
data_plot['type'] = data_plot.type.str.replace(' \(% of male labor force\)','')
data_plot['type'] = data_plot.type.str.replace(' \(% of female labor force\)','')
data_plot = data_plot.pivot_table(values='Value',index='CountryName',columns='type')

```

##### Plot
```python
sns.set(style="white",
        palette="pastel",
        rc={'figure.figsize':(7,5),
            'figure.dpi':500})

sns.lmplot(x = 'female', y = 'male', data = data_plot, fit_reg=False, x_jitter=1.5, y_jitter=1.5)
plt.xlim((0,40))
plt.ylim((0,40))
plt.grid(color='black', linestyle='-.', linewidth=0.25)
plt.title('Unemployment (% of total)',)
plt.savefig('./plots/02.scatter.png',orientation='landscape',dpi=500);
```


![png](\images\blog\distributions\02.scatter.png){: .center-image height="500px" width="750px"}


### Density plot:

##### Data Prep
```python
selected_indicators = ['Mortality rate, adult, female (per 1,000 female adults)',
                       'Mortality rate, adult, male (per 1,000 male adults)']

countries = data_countries.CountryCode[data_countries.Region!=''].unique()
condition = data_indicators.IndicatorName.isin(selected_indicators)

data_plot = data_indicators.loc[condition,:]
condition = data_plot.CountryCode.isin(countries)
data_plot = data_plot.loc[condition,:]
data_plot.sort_values(['CountryName','IndicatorName','Year'], inplace=True)

data_plot = data_plot.groupby(['CountryName','IndicatorName'], as_index=False).last()
data_plot[['feature','type']] = data_plot['IndicatorName'].str.split(', adult, ',expand=True)
data_plot.reset_index(inplace=True, drop=True)
data_plot['type'] = data_plot.type.str.replace(' \(per 1,000 female adults\)','')
data_plot['type'] = data_plot.type.str.replace(' \(per 1,000 male adults\)','')
data_plot = data_plot.pivot_table(values='Value',index='CountryName',columns='type')
```

##### Plot
```python
sns.set(style="white",
        palette="pastel",
        color_codes=True,
        rc={
            'figure.figsize':(10,6),
            'figure.dpi':200
           })

sns.kdeplot(data_plot.male, color='red')
sns.kdeplot(data_plot.female, color='blue')
plt.grid(color='black',linestyle='-.', linewidth=0.25)
plt.title('Mortality rate')
plt.ylim((0,0.006))
plt.xlim((-100,700))
plt.savefig('./03.density.png');
```


![png](\images\blog\distributions\03.density.png){: .center-image height="500px" width="750px"}


### Boxplot:

##### Data prep
```python
selected_indicators = ['Merchandise trade (% of GDP)']

countries = data_countries.CountryCode[data_countries.Region!=''].unique()
condition = data_indicators.IndicatorName.isin(selected_indicators)

data_plot = data_indicators.loc[condition,:]
condition = data_plot.CountryCode.isin(countries)
data_plot = data_plot.loc[condition,:]
data_plot.sort_values(['CountryName','IndicatorName','Year'], inplace=True)

data_plot = data_plot.groupby(['CountryName','IndicatorName'], as_index=False).last()
data_plot.reset_index(inplace=True, drop=True)
data_plot['Region'] = data_plot.merge(right=data_countries,on='CountryCode',how='left')['Region']
```

##### Plot
```python
scolumns_order = sort(data_plot.Region.unique())

sns.set(style="white",
        palette="pastel",
        color_codes=True,
        rc={
            'figure.figsize':(10,6),'figure.dpi':200
           })

sns.boxplot(x='Region',
            y='Value',
            palette='autumn',
            order=columns_order,
            width=0.4,
            fliersize=3,
            data=data_plot);
plt.grid(color='black',linestyle='-.', linewidth=0.25)
plt.xticks(rotation=30)
plt.title('Merchandise trade')
plt.ylabel('% of GDP');
plt.savefig('./04.boxplot.png');
```


![png](\images\blog\distributions\04.boxplot.png){: .center-image height="500px" width="950px"}


### Violin plot:


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


![png](\images\blog\distributions\output_13_0.png){: .center-image height="500px" width="750px"}


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


![png](\images\blog\distributions\output_15_0.png){: .center-image height="500px" width="750px"}



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


![png](\images\blog\distributions\output_16_0.png){: .center-image height="500px" width="750px"}


### Heatmap:


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


![png](\images\blog\distributions\output_18_0.png){: .center-image height="600px" width="1000px"}


### Rugs:

##### Data prep
```python
selected_indicators = ['Merchandise trade (% of GDP)']

countries = data_countries.CountryCode[data_countries.Region!=''].unique()
condition = data_indicators.IndicatorName.isin(selected_indicators)

data_plot = data_indicators.loc[condition,:]
condition = data_plot.CountryCode.isin(countries)
data_plot = data_plot.loc[condition,:]
data_plot.sort_values(['CountryName','IndicatorName','Year'], inplace=True)

data_plot = data_plot.groupby(['CountryName','IndicatorName'], as_index=False).last()
data_plot.reset_index(inplace=True, drop=True)
data_plot['Region'] = data_plot.merge(right=data_countries,on='CountryCode',how='left')['Region']
```

##### Plot
```python
columns_order = sort(data_plot.Region.unique())

sns.set(style="white",
        palette="pastel",
        color_codes=True,
        rc={
            'figure.figsize':(12,8),'figure.dpi':500
           })

g = sns.FacetGrid(data_plot,
                  col="Region",
                  col_wrap=4,
                  col_order=columns_order,subplot_kws={'ylim':(0,0.02)})
g.map(sns.distplot, "Value", hist=False, rug=True);
plt.savefig('./plots/07.rugplot.png', dpi=500, bbox_inches='tight');
```

![png](\images\blog\distributions\07.rugplot.png){: .center-image height="600px" width="1000px"}
