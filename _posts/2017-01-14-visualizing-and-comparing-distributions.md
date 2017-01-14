---
layout: single
title: "seaborn distribution plots - reading development indicators across countries"
date: 2017-01-14T15:39:55-04:00
last_modified_at: 2023-06-10T10:00:00-04:00
categories: [blog]
tags: [data, distribution, visualization, seaborn]
excerpt: "Seven seaborn plots on one column of the World Development Indicators dataset, and how to decide which one to reach for."
redirect_from:
  - /blog/visualizing-&-comparing-distributions/
  - /blog/visualizing-%26-comparing-distributions/
header:
  overlay_image: /images/blog/headers/visualizing-and-comparing-distributions.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/visualizing-and-comparing-distributions.jpg
---

Looking at the shape of a column is a different way of seeing it than reading its mean. The mean of life expectancy gives you one number. The shape might be one tight cluster of countries, or it might be two separate humps, a richer world and a poorer one, that happen to average out to the same place. I find I keep wanting to see which it is. So this post is me working through the plots that let you look.

It is a tour of seven seaborn plots. Each one reads the same kind of column from the World Development Indicators dataset: one value per country for some indicator. Each is a different lens on that column. The histogram asks whether the spread is one hump or two. The scatter is for two numbers per country that might move together. The density trades the histogram's bins for a smooth curve. The box and violin compare the shape once you split by region. The heatmap collapses a whole matrix into a grid of color. The rug goes all the way back down to the raw ticks. Seven views of the same data, and each one foregrounds a different feature of the shape.

<div class="notice--info" markdown="1">
**Updated 2023.** I wrote this in 2017 against seaborn's old `distplot` helper. seaborn deprecated `distplot` in v0.11 (2020) and folded its job into `histplot` and `displot`, so I re-ran the notebook on seaborn 0.12 and the code blocks below use `histplot(kde=True)`, `kdeplot`, and `rugplot`, passing their data by keyword. The plots and the reasoning are unchanged from the original.
</div>

<figure>
<svg viewBox="0 0 640 280" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" aria-labelledby="lenses-title lenses-desc">
<title id="lenses-title">One column of values fanning out into seven plot types</title>
<desc id="lenses-desc">A vertical strip of dots representing one value per country, with arrows fanning out to seven small plot glyphs: histogram bars, a scatter cloud, a density bump, a box, a violin, a heatmap grid, and rug ticks.</desc>
<g fill="currentColor" stroke="none">
<circle cx="60" cy="60" r="4"/>
<circle cx="60" cy="85" r="4"/>
<circle cx="60" cy="110" r="4"/>
<circle cx="60" cy="135" r="4"/>
<circle cx="60" cy="160" r="4"/>
<circle cx="60" cy="185" r="4"/>
<circle cx="60" cy="210" r="4"/>
</g>
<text x="60" y="38" fill="currentColor" font-size="13" text-anchor="middle">one column</text>
<g stroke="currentColor" fill="none" opacity="0.4" stroke-width="1.5">
<path d="M80 135 C 160 40, 200 40, 240 45"/>
<path d="M80 135 C 160 90, 200 90, 240 90"/>
<path d="M80 135 C 160 130, 200 130, 240 132"/>
<path d="M80 135 C 160 135, 200 175, 240 178"/>
<path d="M80 135 C 160 180, 200 222, 240 224"/>
</g>
<g stroke="currentColor" fill="none" stroke-width="1.5">
<g transform="translate(300,30)">
<rect x="0" y="14" width="8" height="16"/>
<rect x="10" y="6" width="8" height="24"/>
<rect x="20" y="0" width="8" height="30"/>
<rect x="30" y="10" width="8" height="20"/>
<rect x="40" y="20" width="8" height="10"/>
<text x="70" y="22" fill="currentColor" stroke="none" font-size="12">histogram</text>
</g>
<g transform="translate(300,78)">
<circle cx="6" cy="20" r="2"/><circle cx="16" cy="10" r="2"/><circle cx="22" cy="24" r="2"/><circle cx="30" cy="6" r="2"/><circle cx="40" cy="16" r="2"/><circle cx="14" cy="26" r="2"/>
<text x="70" y="20" fill="currentColor" stroke="none" font-size="12">scatter</text>
</g>
<g transform="translate(300,122)">
<path d="M0 28 C 16 28, 16 2, 24 2 C 32 2, 32 28, 48 28"/>
<text x="70" y="22" fill="currentColor" stroke="none" font-size="12">density</text>
</g>
<g transform="translate(300,166)">
<line x1="24" y1="0" x2="24" y2="6"/>
<rect x="8" y="6" width="32" height="18"/>
<line x1="8" y1="15" x2="40" y2="15"/>
<line x1="24" y1="24" x2="24" y2="30"/>
<text x="70" y="20" fill="currentColor" stroke="none" font-size="12">box</text>
</g>
<g transform="translate(300,206)">
<path d="M24 2 C 8 8, 8 22, 24 28 C 40 22, 40 8, 24 2 Z"/>
<line x1="24" y1="2" x2="24" y2="28" stroke-dasharray="2 2" opacity="0.5"/>
<text x="70" y="18" fill="currentColor" stroke="none" font-size="12">violin</text>
</g>
<g transform="translate(470,78)">
<rect x="0" y="0" width="48" height="36"/>
<line x1="16" y1="0" x2="16" y2="36" opacity="0.5"/><line x1="32" y1="0" x2="32" y2="36" opacity="0.5"/>
<line x1="0" y1="12" x2="48" y2="12" opacity="0.5"/><line x1="0" y1="24" x2="48" y2="24" opacity="0.5"/>
<text x="60" y="22" fill="currentColor" stroke="none" font-size="12">heatmap</text>
</g>
<g transform="translate(470,160)">
<line x1="2" y1="0" x2="2" y2="14"/><line x1="10" y1="0" x2="10" y2="14"/><line x1="14" y1="0" x2="14" y2="14"/><line x1="24" y1="0" x2="24" y2="14"/><line x1="30" y1="0" x2="30" y2="14"/><line x1="42" y1="0" x2="42" y2="14"/>
<text x="60" y="12" fill="currentColor" stroke="none" font-size="12">rug</text>
</g>
</g>
</svg>
<figcaption>Every figure in this post reads the same kind of column, one value per country, through a different lens. The plot you reach for is a choice about which feature of the shape you want to make visible.</figcaption>
</figure>

## One table, one prep pattern

The whole post runs off a single sqlite file. The data behind it is the World Bank's World Development Indicators, packaged into one `database.sqlite`. You connect once, pull three tables, and you are set.

<style>
a.btn-soft {
  display: inline-block;
  margin: 0 0.5rem 0.55rem 0;
  padding: 0.5em 1.05em;
  font-size: 0.92rem;
  font-weight: 500;
  line-height: 1.3;
  color: inherit;
  text-decoration: none;
  border-radius: 8px;
  border: 1px solid rgba(128, 128, 128, 0.30);
  border: 1px solid color-mix(in srgb, currentColor 22%, transparent);
  background: rgba(128, 128, 128, 0.06);
  background: color-mix(in srgb, currentColor 5%, transparent);
  transition: background-color .18s ease, border-color .18s ease, color .18s ease;
}
a.btn-soft:hover {
  text-decoration: none;
  border-color: rgba(128, 128, 128, 0.55);
  border-color: color-mix(in srgb, currentColor 42%, transparent);
  background: rgba(128, 128, 128, 0.12);
  background: color-mix(in srgb, currentColor 11%, transparent);
}
a.btn-soft--primary {
  font-weight: 600;
  border-color: rgba(128, 128, 128, 0.55);
  border-color: color-mix(in srgb, currentColor 42%, transparent);
}
</style>

That `database.sqlite` is the classic Kaggle packaging of the World Bank data, the same file the code below queries, so the seven plots reproduce from the source rather than from my copy of it.

[Grab the World Development Indicators dataset on Kaggle](https://www.kaggle.com/datasets/kaggle/world-development-indicators){: .btn-soft .btn-soft--primary}

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

`data_indicators` is a long table. There is one row per `CountryName` x `IndicatorName` x `Year`, with the number itself in `Value`. That long shape is the substrate for everything below.

<figure>
<svg viewBox="0 0 640 220" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" aria-labelledby="long-title long-desc">
<title id="long-title">Reducing the long indicators table to one row per country</title>
<desc id="long-desc">A tall narrow table with columns CountryName, IndicatorName, Year and Value, an arrow labeled filter real countries, keep last year, group pointing to a short one-row-per-country table, then an arrow to a small distribution glyph.</desc>
<g stroke="currentColor" fill="none" stroke-width="1.5">
<rect x="20" y="30" width="150" height="160"/>
<line x1="20" y1="60" x2="170" y2="60"/>
<line x1="20" y1="90" x2="170" y2="90"/>
<line x1="20" y1="120" x2="170" y2="120"/>
<line x1="20" y1="150" x2="170" y2="150"/>
</g>
<g fill="currentColor" stroke="none" font-size="10">
<text x="28" y="48">CountryName</text>
<text x="28" y="78">IndicatorName</text>
<text x="28" y="108">Year</text>
<text x="28" y="138">Value</text>
<text x="28" y="172">...</text>
</g>
<text x="95" y="210" fill="currentColor" font-size="11" text-anchor="middle">long table</text>
<g stroke="currentColor" fill="none" stroke-width="1.5">
<path d="M180 110 L 290 110"/>
<path d="M282 104 L 290 110 L 282 116"/>
</g>
<text x="235" y="96" fill="currentColor" font-size="9.5" text-anchor="middle">filter real countries,</text>
<text x="235" y="130" fill="currentColor" font-size="9.5" text-anchor="middle">keep last year, group</text>
<g stroke="currentColor" fill="none" stroke-width="1.5">
<rect x="300" y="70" width="140" height="80"/>
<line x1="300" y1="100" x2="440" y2="100"/>
<line x1="370" y1="70" x2="370" y2="150"/>
</g>
<g fill="currentColor" stroke="none" font-size="10">
<text x="312" y="90">country</text>
<text x="382" y="90">Value</text>
<text x="312" y="125">...</text>
<text x="382" y="125">...</text>
</g>
<text x="370" y="170" fill="currentColor" font-size="11" text-anchor="middle">one row per country</text>
<g stroke="currentColor" fill="none" stroke-width="1.5">
<path d="M450 110 L 530 110"/>
<path d="M522 104 L 530 110 L 522 116"/>
<path d="M545 140 C 575 140, 575 80, 590 80 C 605 80, 605 140, 625 140"/>
</g>
<text x="585" y="160" fill="currentColor" font-size="11" text-anchor="middle">shape</text>
</svg>
<figcaption>The same boilerplate runs before every plot: keep only rows whose country has a real region, take each country's most recent year, then group down to one value per country. Name it once and the per-section data prep is just a change of indicator string.</figcaption>
</figure>

Most of the data-prep blocks below are the same four moves. You filter `data_indicators` down to the indicator strings you want. You drop aggregate rows by keeping only countries that carry a real `Region`. You sort and take each country's last available year. You group to one row per country. Where a section needs two indicators side by side, it pivots them into columns at the end. I kept the blocks verbatim so each one is runnable on its own. Once you have seen the histogram prep you have seen the pattern, and you can skim the rest for the one line that changes: `selected_indicators`.

## Histogram: one hump or two?

Start with the simplest lens. You have one column of values, life expectancy at birth. A histogram shows whether that spread is one cluster or several.

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

```python
nbins = 15
sns.set(style="white",
        palette="pastel",
        color_codes=True,
        rc={'figure.figsize':(12,8),
            'figure.dpi':500})

sns.histplot(data_plot.Value[data_plot.type=='female (years)'], bins=nbins, kde=True)
sns.histplot(data_plot.Value[data_plot.type=='male (years)'], bins=nbins, kde=True)
sns.histplot(data_plot.Value[data_plot.type=='total (years)'], bins=nbins, kde=True)
plt.legend(['Female', 'Male', 'Total'], bbox_to_anchor=(1.12,1.04))
plt.xlim((25,100))
plt.grid(color='black',linestyle='-.',linewidth=0.25)
plt.title('Life expectancy at birth ( In years )')
```

This draws three overlaid histograms, female, male and total, each with its KDE curve laid on top via `kde=True`. The `nbins = 15` and `plt.xlim((25,100))` are the two knobs that matter. The bin count decides how coarse the shape reads, and the x-limit frames the same 25-to-100 range for all three so they overlay on one axis. One thing to watch with the overlay is occlusion. The code sets no `alpha` and no `element='step'`, so the filled bars can hide each other where they stack, and the KDE curves do most of the work of telling the three apart.

<figure>
<img src="/images/blog/distributions/01.histogram.png" alt="Three overlaid histograms with KDE curves showing the spread of female, male, and total life expectancy at birth across countries">
<figcaption>Life expectancy at birth, female, male and total, overlaid across countries. The mass sits high, near 70 to 80 years, with a long tail running down toward 50. The bin count sets how coarse that shape reads.</figcaption>
</figure>

## Scatter: when one number per country becomes two

A histogram reads one column. The moment you have two numbers per country, the question changes. It stops being where the mass sits and becomes whether the two move together, and that is a scatter. Here it is male versus female unemployment.

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

`fit_reg=False` keeps `lmplot` from drawing a regression line, because a fitted slope would average the countries into one trend and smooth away the cloud this plot exists to show. The `x_jitter=1.5` and `y_jitter=1.5` nudge overlapping points apart, so a stack of countries at the same rate does not hide behind one marker. `plt.xlim((0,40))` with `plt.ylim((0,40))` puts both axes on the same scale, which is what makes the diagonal readable. Points on the line are countries where male and female unemployment match, and the spread off it is the gap between them.

<figure>
<img src="/images/blog/distributions/02.scatter.png" alt="Scatter plot of male versus female unemployment rate per country on matched axes from zero to forty percent">
<figcaption>Male against female unemployment, one point per country. Two numbers per country need two axes, and the diagonal does the work: how far a country sits off the 45-degree line is how far its two rates disagree.</figcaption>
</figure>

## Density: a histogram without the bin argument

Go back to one column, but drop the bins. A kernel density estimate is the histogram's smooth cousin. You no longer have a bin count to defend, but you pay for it with smoothness the data never had. The bin count reappears as the bandwidth, hidden inside seaborn's default Scott rule rather than spelled out, so the lens still has a knob, one you usually never touch. Here it is adult mortality, male and female, side by side.

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

```python
sns.set(style="white",
        palette="pastel",
        color_codes=True,
        rc={
            'figure.figsize':(10,6),
            'figure.dpi':200
           })

sns.kdeplot(data=data_plot, x='male', color='red')
sns.kdeplot(data=data_plot, x='female', color='blue')
plt.grid(color='black',linestyle='-.', linewidth=0.25)
plt.title('Mortality rate')
plt.ylim((0,0.006))
plt.xlim((-100,700))
plt.savefig('./03.density.png');
```

This draws two `kdeplot` curves, male in red and female in blue. The `plt.ylim((0,0.006))` and `plt.xlim((-100,700))` fix the frame so the two densities are comparable. The negative left edge is the tell. The kernel happily smears probability mass below zero, into mortality rates that cannot exist. That is the cost of the lens: it buys you a clean curve by inventing a little continuity at the edges.

<figure>
<img src="/images/blog/distributions/03.density.png" alt="Two kernel density curves comparing adult mortality rate distributions for male and female populations across countries">
<figcaption>Adult mortality, male against female, as kernel density curves. The smoothing frees you from a bin count but pays for it with a tail the raw data never had, including mass smeared left of zero where no mortality rate can live.</figcaption>
</figure>

## Box and violin: now group by region

So far every plot has read one ungrouped column. The next question is whether the shape changes when you split countries into groups. Two lenses answer it. A box plot gives you the five-number summary per group, the median line and the quartile box with its whiskers, and throws the rest away. A violin gives you the box plus the kernel shape it discarded.

<figure>
<svg viewBox="0 0 480 240" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" aria-labelledby="boxviolin-title boxviolin-desc">
<title id="boxviolin-title">A box plot beside a violin of the same distribution</title>
<desc id="boxviolin-desc">On the left a box plot glyph with median line, quartile box and whiskers. On the right a violin glyph of the same outline, with a faint dashed box overlaid to show the violin is the box plus the kernel shape it keeps.</desc>
<g stroke="currentColor" fill="none" stroke-width="1.5">
<line x1="120" y1="30" x2="120" y2="60"/>
<rect x="90" y="60" width="60" height="90"/>
<line x1="90" y1="105" x2="150" y2="105"/>
<line x1="120" y1="150" x2="120" y2="190"/>
<line x1="105" y1="30" x2="135" y2="30"/>
<line x1="105" y1="190" x2="135" y2="190"/>
</g>
<text x="120" y="220" fill="currentColor" font-size="13" text-anchor="middle">box: five numbers</text>
<g stroke="currentColor" fill="none" stroke-width="1.5">
<path d="M360 30 C 320 55, 320 80, 350 105 C 322 130, 322 165, 360 190 C 398 165, 398 130, 370 105 C 400 80, 400 55, 360 30 Z"/>
</g>
<g stroke="currentColor" fill="none" stroke-width="1.2" opacity="0.45" stroke-dasharray="3 3">
<rect x="345" y="60" width="30" height="90"/>
<line x1="345" y1="105" x2="375" y2="105"/>
<line x1="360" y1="30" x2="360" y2="60"/>
<line x1="360" y1="150" x2="360" y2="190"/>
</g>
<text x="360" y="220" fill="currentColor" font-size="13" text-anchor="middle">violin: box + shape</text>
</svg>
<figcaption>The box and the violin describe the same distribution. The box keeps five numbers and the violin keeps the whole silhouette, with the dashed box drawn inside to show what it still carries. If a group hides two clusters, the box will look identical to a single-hump group and the violin will not.</figcaption>
</figure>

First the box. This one is merchandise trade as a share of GDP, split by world region.

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

```python
columns_order = sorted(data_plot.Region.unique())

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

`width=0.4` keeps the boxes slim so the region labels breathe, and `fliersize=3` shrinks the outlier dots that would otherwise crowd the high-trade regions. You get one box per region, each a five-number summary. The comparison you get for free is which regions trade more relative to their size and which spread wider. The `sorted(data_plot.Region.unique())` is deliberate. `%pylab inline` pulls `numpy.sort` into the namespace as a bare `sort`, so I reach for the builtin `sorted` to be explicit about fixing the column order across every grouped plot below.

<figure>
<img src="/images/blog/distributions/04.boxplot.png" alt="Box plot of merchandise trade as a percent of GDP for each world region with outliers shown as small dots">
<figcaption>Merchandise trade as a share of GDP, one box per region. Region medians land near 50 percent, and every region carries a few high-trade outliers, some stretched past 200 percent of GDP.</figcaption>
</figure>

Now the violin, which keeps the shape the box drops. This one also splits each region in half, gaseous against liquid CO2 emissions.

```python
selected_indicators = [ 'CO2 emissions from gaseous fuel consumption (% of total)',
                        'CO2 emissions from liquid fuel consumption (% of total)']

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

```python
import matplotlib.patches as mpatches

columns_order = sorted(data_plot.Region.unique())

sns.set(style="white",
        palette="pastel",
        color_codes=True,
        rc={
            'figure.figsize':(16,10),'figure.dpi':250
           })
sns.violinplot(x ='Region',
               y='Value',
               hue='IndicatorName',
               linewidth=0.25,
               inner="quart",
               palette={"CO2 emissions from gaseous fuel consumption (% of total)": "y",
                        "CO2 emissions from liquid fuel consumption (% of total)": "b"},
               data=data_plot,
               split=True)
plt.grid(color='black',linestyle='-.', linewidth=0.25)
plt.title('CO$_2$ emission')
plt.ylabel('% of total')

gas_patch = mpatches.Patch(color='yellow', label='Gaseous',alpha=0.5)
liquid_patch = mpatches.Patch(color='skyblue', label='Liquid')
plt.legend(handles=[gas_patch, liquid_patch], bbox_to_anchor=(0.2, 0.99), fontsize='x-large')
plt.savefig('./plots/05.violinplot.png', dpi=250, bbox_inches='tight');
```

`split=True` with `inner="quart"` is the trick that earns the violin its keep. Gaseous goes on one side, liquid on the other, with quartile lines drawn inside each half so you still get the box-style summary without a second plot. Where a region's silhouette bulges in two places, you are looking at a cluster a box would have flattened into one.

<figure>
<img src="/images/blog/distributions/05.violinplot.png" alt="Split violin plot per region comparing the distribution of gaseous and liquid CO2 emissions as a percent of total">
<figcaption>Gaseous against liquid CO2 emissions, one split violin per region. Each half carries the quartile lines a box would give you, wrapped in the density curve a box drops.</figcaption>
</figure>

## Heatmap: a whole matrix at once

The plots so far read a column, or a column split into groups. A heatmap is for when the data is already a matrix, region by trade destination, and you have run out of position axes. Color becomes the only channel left.

```python
selected_indicators_export = [
    'Merchandise exports to developing economies in East Asia & Pacific (% of total merchandise exports)',
    'Merchandise exports to developing economies in Latin America & the Caribbean (% of total merchandise exports)',
    'Merchandise exports to developing economies in Middle East & North Africa (% of total merchandise exports)',
    'Merchandise exports to developing economies in South Asia (% of total merchandise exports)',
    'Merchandise exports to developing economies in Sub-Saharan Africa (% of total merchandise exports)',
    'Merchandise exports to developing economies outside region (% of total merchandise exports)',
    'Merchandise exports to developing economies within region (% of total merchandise exports)',
    'Merchandise exports to economies in the Arab World (% of total merchandise exports)',
    'Merchandise exports to high-income economies (% of total merchandise exports)'
]

selected_indicators_imports = [
    'Merchandise imports from developing economies in East Asia & Pacific (% of total merchandise imports)',
    'Merchandise imports from developing economies in Latin America & the Caribbean (% of total merchandise imports)',
    'Merchandise imports from developing economies in Middle East & North Africa (% of total merchandise imports)',
    'Merchandise imports from developing economies in South Asia (% of total merchandise imports)',
    'Merchandise imports from developing economies in Sub-Saharan Africa (% of total merchandise imports)',
    'Merchandise imports from developing economies outside region (% of total merchandise imports)',
    'Merchandise imports from developing economies within region (% of total merchandise imports)',
    'Merchandise imports from economies in the Arab World (% of total merchandise imports)',
    'Merchandise imports from high-income economies (% of total merchandise imports)'
]

countries = data_countries.CountryCode[data_countries.Region!=''].unique()
condition = data_indicators.IndicatorName.isin(selected_indicators_export)
data_plot = data_indicators.loc[condition,:]
condition = data_plot.CountryCode.isin(countries)
data_plot = data_plot.loc[condition,:]
data_plot.sort_values(['CountryName','IndicatorName','Year'], inplace=True)
data_plot = data_plot.groupby(['CountryName','IndicatorName'], as_index=False).last()
data_plot.reset_index(inplace=True, drop=True)
data_plot['Region'] = data_plot.merge(right=data_countries,on='CountryCode',how='left')['Region']
data_export = data_plot.pivot_table(values='Value',columns='Region',index='IndicatorName')


countries = data_countries.CountryCode[data_countries.Region!=''].unique()
condition = data_indicators.IndicatorName.isin(selected_indicators_imports)
data_plot = data_indicators.loc[condition,:]
condition = data_plot.CountryCode.isin(countries)
data_plot = data_plot.loc[condition,:]
data_plot.sort_values(['CountryName','IndicatorName','Year'], inplace=True)
data_plot = data_plot.groupby(['CountryName','IndicatorName'], as_index=False).last()
data_plot.reset_index(inplace=True, drop=True)
data_plot['Region'] = data_plot.merge(right=data_countries,on='CountryCode',how='left')['Region']
data_import = data_plot.pivot_table(values='Value',columns='Region',index='IndicatorName')
```

```python
sns.set(style="white",
        color_codes=True,
        rc={
            'figure.figsize':(20,8),
            'figure.dpi':250
           })
fig, (imports, exports) = plt.subplots(1, 2, sharex=True)

im1 = sns.heatmap(data_import.loc[:,xlabels],
                  ax=imports,
                  center=50,
                  cbar=False,
                  cmap="YlGnBu")
imports.set_yticklabels(ylabels)
imports.set_ylabel('')
imports.set_xlabel('')
imports.set_title('Imports :');

im2 = sns.heatmap(data_export.loc[:,xlabels],
                  ax=exports,
                  center=50,
                  yticklabels=False,
                  cbar=False,
                  cmap="YlGnBu")
exports.set_ylabel('')
exports.set_xlabel('')
exports.set_title('Exports :');
fig.subplots_adjust(wspace=0.05, hspace=0)

mappable = im1.get_children()[0]
fig.colorbar(mappable, ax = [imports,exports],orientation = 'vertical')
plt.savefig('./plots/06.heatmap.png', dpi=250, bbox_inches='tight');
```

Two heatmaps share a colorbar, imports on the left and exports on the right. `center=50` anchors the colormap so the eye reads above and below the halfway mark, and `wspace=0.05, hspace=0` pulls the panels tight so they read as one figure. One line here needs a stub before it runs. This block references `xlabels` and `ylabels` that I defined further up in the original notebook and never carried into the post. They are just the column order and the row labels for the two pivot tables, so the quickest stub is to derive them from what is already in scope:

```python
xlabels = sorted(data_import.columns)
ylabels = [name.split(' (% ')[0] for name in data_import.index]
```

<figure>
<img src="/images/blog/distributions/06.heatmap.png" alt="Side by side heatmaps of merchandise imports and exports by region sharing one color bar centered at fifty">
<figcaption>Merchandise imports and exports by region, two matrices sharing a colorbar. When the data is region by destination, position is spent on both axes and color carries the value. A heatmap trades precise reading for the ability to scan a whole grid at a glance.</figcaption>
</figure>

## Rug: the raw data under everything

Close the tour by going all the way back down. Every plot above binned or smoothed the column into a summary first. The rug does none of that. It draws every country as a single tick, the raw marks the other six plots were built on top of.

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

```python
columns_order = sorted(data_plot.Region.unique())

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
g.map(sns.kdeplot, "Value")
g.map(sns.rugplot, "Value");
plt.savefig('./plots/07.rugplot.png', dpi=500, bbox_inches='tight');
```

`col_wrap=4` lays the regions out four to a row as a small-multiples grid. `subplot_kws={'ylim':(0,0.02)}` pins every panel to the same y-range so the densities are comparable across regions, and each panel layers a `kdeplot` curve over a `rugplot` of the actual ticks. The rug is the reality check on the curve above it. Where the density says smooth bump, the rug shows you whether that bump rests on twenty countries or three. And that reality check is also the limit of the whole tour. None of these seven plots test anything. They form hypotheses, they do not confirm them. A violin that bulges twice is a reason to go check, not a finding. What the rug keeps honest is the eye that the other six lenses train: glance at the raw ticks whenever a smooth curve looks too clean to trust.

<figure>
<img src="/images/blog/distributions/07.rugplot.png" alt="Small multiples of merchandise trade per region, each panel a density curve over a rug of individual country ticks">
<figcaption>Merchandise trade by region, a density curve over a rug of individual country ticks, faceted four to a row. In North America and South Asia the curve rests on only two or three ticks; in Europe and Sub-Saharan Africa it rests on dozens.</figcaption>
</figure>

## Which one to reach for

The heuristic is short. One column, reach for a histogram or a density. Two numbers per country, a scatter. A column that splits into groups, a box for the fast comparison or a violin when you suspect the box is hiding a second hump. A full matrix, a heatmap.

The deeper point is that the plot you pick decides what you are allowed to notice. A box can never show you bimodality. A density can never tell you it rests on three points. The mean cannot tell you anything at all. So the move is to look through more than one lens before you believe what you see.
