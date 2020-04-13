from urllib.request import urlopen
import json
import plotly.express as px
from wiki_pl_scraper import crawl_wiki_pl
from IPython import embed

# get most recent data on portugal
df_pl_data = crawl_wiki_pl()
print(df_pl_data)

# get a geojson for PL
pl_geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/portugal.geojson"
geojson_pl = None
with urlopen(pl_geojson_url) as r:
    geojson_pl = json.load(r)

embed()

fig = px.choropleth(df_pl_data, geojson=geojson_pl, color='An\xadzahl',
                    locations=geojson_pl.index, featureidkey="id",
                    projection="mercator")
fig.update_geos(fitbounds="locations", visible=True)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()
