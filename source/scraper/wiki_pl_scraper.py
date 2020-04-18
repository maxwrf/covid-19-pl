import os
from bs4 import BeautifulSoup
import pandas as pd
import requests
from datetime import date


headers = {
    'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)\
                     AppleWebKit/537.36 (KHTML, like Gecko)\
                     Chrome/39.0.2171.95 Safari/537.36"
}

wiki_url = 'https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Portugal'


def crawl_wiki_pl(save_csv=True, no_total=True):
    try:
        today = date.today()
        dt = today.strftime("%Y-%m-%d")
        d = pd.read_csv('data/pl_regions.csv').iloc[-2, 0]

        if d == dt:
            return

    except BaseException:
        pass

    page = requests.get(wiki_url, headers=headers).text
    soup = BeautifulSoup(page, 'lxml')
    table = soup.find_all('table')[4]

    # unstack column index
    df_by_region = pd.read_html(str(table))[0]
    df_by_region.columns = ['%s%s' % (a, '|%s' % b if b != a else '')
                            for a, b in df_by_region.columns]

    # remove notes
    df_by_region = df_by_region.iloc[:-1, :]

    # save to hard drive
    if save_csv:
        df_by_region.to_csv(os.getcwd() + '/data/pl_regions.csv', index=False)


if __name__ == '__main__':
    crawl_wiki_pl()
