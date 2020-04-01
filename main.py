from urllib.request import urlopen
import json
import plotly.express as px
from scraper import crawl_wiki_pl
from IPython import embed

# get rki data
df_pl_data = crawl_wiki_pl()
most_recent = df_pl_data.loc['Total', :]

# get a geojson for PL
pl_geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/portugal.geojson"
geojson_pl = None
with urlopen(pl_geojson_url) as r:
    geojson_pl = json.load(r)

embed()

fig = px.choropleth(df_rki_data, geojson=geojson_germany, color='An\xadzahl',
                    locations=df_rki_data.index, featureidkey="id",
                    projection="mercator")
fig.update_geos(fitbounds="locations", visible=True)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()
