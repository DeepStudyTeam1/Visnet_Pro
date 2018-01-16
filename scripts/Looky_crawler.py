from bs4 import BeautifulSoup
import urllib.request

OUTPUT_FILE_NAME = 'output.pkl'

URL = 'http://www.afit.co.kr/m/product.html?branduid=86961&xcode=001&mcode=001&scode=&type=X&sort=order&cur_code=001001&GfDT=bW93U10%3D'


# 크롤링 함수
def get_text (URL, shop):
    if shop == 'afit' :
        source_code_from_URL = urllib.request.urlopen (URL)
        soup = BeautifulSoup (source_code_from_URL, 'lxml', from_encoding='utf-8')
        for link in soup.find_all ('a'):
            print (link.get ('href'))

    return


# 메인 함수
def main ():
    get_text (URL, 'afit')


if __name__ == '__main__':
    main ()

