from bs4 import BeautifulSoup
import urllib.request

OUTPUT_FILE_NAME = 'output.pkl'

URL = 'http://www.bymono.com/product/detail.html?product_no=22953&cate_no=194&display_group=1'


# 크롤링 함수
def get_image_url (URL, shop):
    source_code_from_URL = urllib.request.urlopen (URL)
    soup = BeautifulSoup (source_code_from_URL, 'lxml', from_encoding='utf-8')
    if shop == 'afit' :
        for something in soup.find_all ('link'):
            some= something.get("rel")
            if some == ["image_src"]:
                print(something.get('href'))
    elif shop == 'allthumb':
        for something in soup.find_all ('img'):
            some = something.get("class")
            print(some)
            if some == ["bigImage"]:
                print(something.get('src'))
    elif shop == 'bechic':
        for something in soup.find_all ('img'):
            some = something.get("class")
            if some == ["BigImage", '']:
                print(something.get('src'))
    elif shop == 'blueforce':
        for something in soup.find_all ('li'):
            some = something.get("class")
            if some == ['origin-img']:
                print("http://www.blueforce.co.kr" + something.a.img.get('src'))
    elif shop == 'bymono':
        for something in soup.find_all ('div'):
            some = something.get("class")
            if some == ['xans-element-', 'xans-product', 'xans-product-image', 'imgArea', '']:
                print("http:" + something.img.get('src'))

    return


# 메인 함수
def main ():
    get_image_url (URL, 'bymono')


if __name__ == '__main__':
    main ()

