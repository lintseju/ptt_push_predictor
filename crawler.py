import argparse
import arrow
import json
import logging
import os
import re
import requests
import time
import traceback

from bs4 import BeautifulSoup
from bs4.element import NavigableString
from tqdm import tqdm

r_mmdd = re.compile(r'.*(\d{2}/\d{2}).*')
r_hhmm = re.compile(r'.*(\d{2}:\d{2}).*')


# https://github.com/zake7749/PTT-Crawler/blob/master/Crawler.py
class PttCrawler:
    # sample usage:
    # crawler = PttCrawler()
    # crawler.crawl(board="Gossiping", start=20000, end=20100)
    root = "https://www.ptt.cc/bbs/"
    main = "https://www.ptt.cc"
    gossip_data = {
        "from": "bbs/Gossiping/index.html",
        "yes": "yes"
    }
    month2int = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

    def __init__(self):
        self.session = requests.session()
        requests.packages.urllib3.disable_warnings()
        self.session.post("https://www.ptt.cc/ask/over18",
                          verify=False,
                          data=self.gossip_data)

    def article_list(self, url, reverse=False):
        """文章內容的生成器
        Args:
            page: 頁面網址
            reverse: 是否從後面開始爬
        Returns:
            文章內容的生成器
        """

        res = self.session.get(url, verify=False)
        soup = BeautifulSoup(res.text, "lxml")

        articles = list(soup.find_all('div', {'class': lambda x: x in ('r-ent', 'r-list-sep')}))
        res = []
        for article in articles:
            if str(article) == '<div class="r-list-sep"></div>':
                break
            try:
                res.append(self.main + article.select(".title")[0].select("a")[0].get("href"))
            except:
                pass  # (本文已被刪除)
        if reverse:
            res = res[::-1]
        return res

    def pages(self, board=None, index_range=None):
        """頁面網址的生成器
        Args:
            board: 看板名稱
            index_range: 文章頁數範圍
        Returns:
            網址的生成器
        """

        target_page = self.root + board + "/index"

        if range is None:
            yield target_page + ".html"
        else:
            for index in index_range:
                yield target_page + str(index) + ".html"

    def parse_article(self, url, mode):
        """解析爬取的文章，整理進dict
        Args:
            url: 欲爬取的PTT頁面網址
            mode: 欲爬取回文的模式。全部(all)、推文(up)、噓文(down)、純回文(normal)
        Returns:
            article: 爬取文章後資料的dict

        """

        # 處理mode標誌
        if mode == 'all':
            mode = 'all'
        elif mode == 'up':
            mode = u'推'
        elif mode == 'down':
            mode = u'噓'
        elif mode == 'normal':
            mode = '→'
        else:
            raise ValueError("mode變數錯誤", mode)

        raw = self.session.get(url, verify=False)
        soup = BeautifulSoup(raw.text, "lxml")

        article = dict()
        try:
            # 取得文章作者與文章標題
            article["Author"] = soup.select(".article-meta-value")[0].contents[0].split(" ")[0]
            article["Title"] = soup.select(".article-meta-value")[2].contents[0]
            date = soup.select(".article-meta-value")[3].contents[0]
            month = date[4:7]
            date = date.replace(month, PttCrawler.month2int[month])
            # hours -8: convert TW time to UTC
            article_time = arrow.get(date[4:].replace('  ', ' '), 'MM D HH:mm:ss YYYY').shift(hours=-8)
            year = '%4d' % article_time.year
            article['time'] = article_time.timestamp
            article['url'] = url
            tokens = url.split('/')
            article['board'] = tokens[tokens.index('bbs') + 1]

            # 取得內文
            content = ""
            for tag in soup.find(id="main-content"):
                if type(tag) is NavigableString:
                    content += tag
                else:
                    if '※ 發信站: 批踢踢實業坊(ptt.cc)' in tag.get_text():
                        break
                    if 'article-metaline' not in tag.attrs.get('class', [''])[0]:
                        content += tag.get_text()
            article["Content"] = content

            # 處理回文資訊
            upvote = 0
            downvote = 0
            novote = 0
            response_list = []

            previous_time = article['time']
            for response_struct in soup.select(".push"):

                # 跳脫「檔案過大！部分文章無法顯示」的 push class
                if "warning-box" not in response_struct['class']:
                    response_dic = dict()
                    response_dic["Content"] = response_struct.select(".push-content")[0].contents[0][1:]
                    response_dic["Vote"] = response_struct.select(".push-tag")[0].contents[0][0]
                    response_dic["User"] = response_struct.select(".push-userid")[0].contents[0]

                    response_time = response_struct.select('.push-ipdatetime')[0].contents[0]
                    mmdd = re.match(r_mmdd, response_time)
                    if mmdd is not None:
                        mmdd = mmdd.group(1)
                    else:
                        response_time = None
                    # 有些版沒有小時：分鐘
                    hhmm = re.match(r_hhmm, response_time)
                    if hhmm is not None:
                        hhmm = hhmm.group(1)
                    else:
                        hhmm = '00:00'

                    if response_time is not None:
                        # warning: 假設推文年份跟文章年份一樣，沒有秒數
                        response_time = arrow.get('%s/%s %s' % (year, mmdd, hhmm), 'YYYY/MM/DD HH:mm')
                        response_dic['ResponseTime'] = response_time.timestamp
                        previous_time = response_dic['ResponseTime']
                    else:
                        # parsing time error, use previous response time
                        response_dic['ResponseTime'] = previous_time

                    # 根據不同的mode去採集response
                    if response_dic["Vote"] == mode or mode == 'all':
                        response_list.append(response_dic)

                        if response_dic["Vote"] == "推":
                            upvote += 1
                        elif response_dic["Vote"] == "噓":
                            downvote += 1
                        else:
                            novote += 1

            article["Responses"] = response_list
            article["UpVote"] = upvote
            article["DownVote"] = downvote
            article["NoVote"] = novote

        except Exception:
            logging.info('Parsing %s error', url)
            logging.info('%s', traceback.print_exc())
            return None

        return article

    @staticmethod
    def output(filename, data):
        """爬取完的資料寫到json文件
        Args:
            filename: json檔的文件路徑
            data: 爬取完的資料
        """

        try:
            with open('Gossiping/' + filename + '.json', 'w') as op:
                json.dump(data, op, ensure_ascii=False)
                logging.info('%s.json saved', filename)
        except Exception as err:
            logging.info('%s.json failed', filename)
            logging.info('error message: %s', err)

    def crawl(self, board, start_ts, end_ts, sleep_time=0.05):
        # FIXME: crawl post between start_ts and end_ts only, do not crawl post not in end_ts
        # todo: add up vote filter
        url = self.root + board + "/index.html"
        raw = self.session.get(url, verify=False)
        soup = BeautifulSoup(raw.text, "lxml")
        pages = soup.find_all('a', href=re.compile(r'/bbs/%s/index\d+.html' % board))
        last_index = 0
        for p in pages:
            last_index = max(int(re.search(r'index(\d+).html', p.get('href')).group(1)), last_index)
        # last_index is previous page of last page, so need to add 1 here
        last_index += 1

        res = []
        done = 0
        while not done:
            url = self.root + board + "/index%d.html" % last_index
            logging.info('Crawling %s', url)
            for article in self.article_list(url, reverse=True):
                article = self.parse_article(article, 'all')
                if article is None:  # parsing error
                    continue
                article['board'] = board
                time.sleep(sleep_time)
                if article['time'] < start_ts:
                    done += 1
                elif article['time'] < end_ts:
                    res.append(article)
                # FIXME: workaround for this weird case: https://www.ptt.cc/bbs/Gossiping/M.1586237539.A.23B.html
                # Fake time on the top of the page
                if done >= 2:
                    break
            last_index -= 1
        return res


def parse_arg():
    parser = argparse.ArgumentParser(description='PTT Data Crawler')
    parser.add_argument('-b', '--board', type=str, default='Boy-Girl', help='Board')
    parser.add_argument('-d', '--date', type=str, default='20200101', help='Start date')
    parser.add_argument('-l', '--length', type=int, default=None, help='Crawling dates')
    return parser.parse_args()


def main():
    args = parse_arg()
    ts = arrow.get(args.date, 'YYYYMMDD').timestamp
    crawler = PttCrawler()
    if args.length is None:
        end_ts = arrow.utcnow().timestamp
    else:
        end_ts = ts + args.length * 86400
    articles = crawler.crawl(args.board, ts, end_ts)
    logging.info('Crawling done, start saving output')
    f = open('data/%s/line.json' % args.board, 'w')
    for i, article in tqdm(enumerate(articles)):
        f.write('%s\n' % json.dumps(article, ensure_ascii=False))
    f.close()
    logging.info('Done')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-2s [%(module)s:%(funcName)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    main()
