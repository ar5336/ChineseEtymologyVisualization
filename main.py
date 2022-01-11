
from bs4 import BeautifulSoup
import urllib.request
import urllib.parse
import urllib.error
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import mechanize
import pinyin
import pinyin.cedict

from googletrans import Translator


IMAGE_WIDTH = 500
IMAGE_HEIGHT = 600


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class glyphNode:
    def __init__(self, glyph, position):
        self.grid_position = position
        self.glyph = glyph
        self.pinyin = pinyin.get(glyph)
        english = pinyin.cedict.translate_word(glyph)
        if english is not None:
            # print(english[0])
            self.english = english[0]
        else:
            self.english = ""
        self.is_explored = False

        self.children_nodes = []

    def add_child_node(self, new_node):
        self.children_nodes.append(new_node)

    def reset_exporation(self):
        self.is_explored = False
        for child_node in self.children_nodes:
            child_node.reset_exploration()

    def render_node(self, image):
        render_scale = 54
        block_size = 1.0

        # print("my glyph is: "+self.glyph)

        image_x1 = int(self.grid_position.x * render_scale + IMAGE_WIDTH / 2)
        image_y1 = int((self.grid_position.y * render_scale) + IMAGE_WIDTH / 4)
        image_x2 = int(image_x1 + render_scale * block_size)
        image_y2 = int(image_y1 + render_scale * block_size)

        rected_im = cv2.rectangle(image, [image_x1, image_y1], [image_x2, image_y2], (255,255,255), thickness=-1)

        cv2.putText(rected_im, self.english, (image_x1 + render_scale, int(image_y1 + render_scale/2)), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1, cv2.LINE_AA)
        # cv2.putText(img, "--- by Silencer", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1, cv2.LINE_AA)

        img_pil = Image.fromarray(rected_im)
        draw = ImageDraw.Draw(img_pil)

        fontpath = "./simsun.ttc"
        font = ImageFont.truetype(fontpath, render_scale-4)



        b, g, r = 0, 0, 0
        draw.text((int(image_x1), int(image_y1)), self.glyph, font=font, fill=(b,g,r))

        img = np.array(img_pil)

        return img

    def scan_children(self):
        derivation_type, child_glyphs = get_components_from_glyph(self.glyph)
        # child_index = 0
        num_children = len(child_glyphs)
        for child_index in range(num_children):
            if num_children == 1:
                self.children_nodes.append(glyphNode(child_glyphs[child_index],
                                                     Point(self.grid_position.x,
                                                           self.grid_position.y + 1)))
            elif num_children == 2:
                self.children_nodes.append(glyphNode(child_glyphs[child_index],
                                                     Point(self.grid_position.x - 2 + 4 * child_index,
                                                           self.grid_position.y + 1)))
            else:
                print("too many children!")


def get_components_from_glyph(glyph):
    # read wikitionary

    # scrape html
    # return composition type and glyphs with optional derivation types

    target_url = "https://en.wiktionary.org/wiki/" + urllib.parse.quote(glyph)

    # web_bytes = urllib.request.urlopen(target_url)
    # web_bytes = urllib.request.urlopen(target_url)
    web_bytes = None
    try:
        web_bytes = urllib.request.urlopen(target_url)
        # break
    except urllib.error.HTTPError as exception:

        print("Oops!  That was no valid number.  Try again...")

    if web_bytes != None:
        english_was_changed = False

        mybytes = web_bytes.read()

        wiki_html = mybytes.decode("utf8")
        web_bytes.close()

        soup = BeautifulSoup(wiki_html, 'html.parser')
        # relevant_entries = soup.find('span', id="Glyph_origin").next_element #.prettify()

        derivation_type = ""
        derivative_glyphs = []

        # test if there's a glyph origin entry
        glyph_origin = soup.find('span', id="Glyph_origin")
        if glyph_origin != None:
            # figure out if phono-semantic or ideographic
            # print(glyph_origin)
            text_entry = glyph_origin.parent.find_next_sibling("p")

            p_text = text_entry.getText()
            words_of_text = p_text.split(" ")
            compund_type = words_of_text[0]
            print(compund_type)
            derivation_type = compund_type

            # all_hani_mentions = text_entry.find_all('i', {"class": "Hani mention"}) + text_entry.find_all('span', {
            #     "class": "Hani"})
            all_links = text_entry.find_all(['i','span'])
            all_glyphs = []
            # print(all_hani)
            for link in all_links:
                a_entry = link.findChildren("a")
                if a_entry is not None and len(a_entry) >= 1:
                    chinese_char = a_entry[0].getText()
                    all_glyphs.append(chinese_char)

            # mentions = all_hani.find([{"class": "Hani mention"},{"class": "Hani"}])
            # print(mentions)
            # all_hani_mentions = text_entry.find_all([('i', {"class": "Hani mention"}), ('span', {"class": "Hani"})])
            # print("number of hani entreis: " + str(len(all_hani_mentions)))

            hani_glyphs = all_glyphs
            # if len(all_hani_mentions) == 3:
            # for hani_mention in all_hani_mentions:
            #     a_entry = hani_mention.findChildren("a")
            #     chinese_char = a_entry[0].getText()
            #
            #     hani_glyphs.append(chinese_char)
            # for hani_glyph in hani_glyphs:
            if len(hani_glyphs) > 0:
                if compund_type == "Variant":
                    derivative_glyphs = [hani_glyphs[0]]
                if compund_type == "Simplified":
                    # print(hani_glyphs)
                    derivative_glyphs = [hani_glyphs[0]]
                elif hani_glyphs[0] == "形聲": #phonosemantic
                    derivative_glyphs = hani_glyphs[1:3]
                elif hani_glyphs[0] == "象形": #pictogram

                    if len(hani_glyphs) >= 2:
                        if hani_glyphs[1] == "女":
                            derivative_glyphs = []
                        else:
                            derivative_glyphs = hani_glyphs[1:3]
                        # derivative_glyphs = hani_glyphs[1::]
                    else:
                        derivative_glyphs = []
                elif hani_glyphs[0] == "會意": #ideogrammic
                    derivative_glyphs = hani_glyphs[1:3]
                    # print(derivative_glyphs)
            # however there is a leftover possibility that the "simplified" infromation is in the table
        # print(relevant_entries)
        return derivation_type, derivative_glyphs
    return "", []


class glyphTree:
    def __init__(self, starting_glyph):
        self.starting_glyph = starting_glyph
        self.root_node = glyphNode(starting_glyph, Point(0, 0))

        # init stuff
        self.fill_tree()

    def fill_tree(self):
        keep_scanning = True
        # current_node = self.root_node
        nodes_to_explore = [self.root_node]
        while len(nodes_to_explore) != 0:
            current_node = nodes_to_explore.pop()
            # scan wiki for component parts
            current_node.scan_children()

            num_children = len(current_node.children_nodes)
            print("the number of children is: "+str(num_children))
            for child_index in range(num_children):
                nodes_to_explore.append(current_node.children_nodes[child_index])
            # production_type, glyphs = get_components_from_glyph(current_node.glyph)
            # # explore all components
            # for i in range(len(glyphs)):
            #     glyph_derived = glyphs[i]
            #
            #     new_glyph_node = glyphNode(glyph_derived, Point(current_node.grid_position.x - 1.5 + 3 * i,
            #                                                     current_node.grid_position.y + 3))
            #     current_node.add_child_node(new_glyph_node)
            #     nodes_to_explore.append(new_glyph_node)
            current_node.is_explored = True
        # fill out the tree by crawling wikitionary

    def tree_to_list(self):
        # flattens the tree of glyphs into a list
        nodes_to_explore = [self.root_node]
        nodes_explored = []
        while len(nodes_to_explore) >= 1:
            focus_node = nodes_to_explore.pop()

            nodes_explored.append(focus_node)
            num_children = len(focus_node.children_nodes)
            if num_children == 0:
                continue
            else:
                for child in focus_node.children_nodes:
                    nodes_to_explore.append(child)
        # for node in nodes_explored:
        #     print(node.glyph+", ")
        return nodes_explored


    def render_tree(self):
        # produce a drawing of the tree of glyphs
        graph_canvas = np.zeros((700, 700, 3), np.uint8)
        # graph_canvas = np.zeros((500, 500, 3))

        tree_list = self.tree_to_list()
        for glyph_node in tree_list:
            graph_canvas = glyph_node.render_node(graph_canvas)
        return graph_canvas

def main():


    # fontpath = "./simsun.ttc" # <== 这里是宋体路径
    fontpath = "./simsun.ttc"
    font = ImageFont.truetype(fontpath, 32)

    # cv2.putText(img,  "--- by Silencer", (200,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv2.LINE_AA)
    translator = Translator()

    english_text = "on time"
    english_was_changed = "true"

    while True:
        img = np.zeros((600, 400, 3), np.uint8)

        b, g, r, a = 255, 255, 255, 0

        if english_was_changed:

            translation = ""
            if len(english_text) >= 1:
                translation = translator.translate(english_text, src='en', dest='zh-CN')

            translated_text = translation.text

            # translated_text =

            if len(translated_text) == 1:
                character_of_interest = translated_text[0]
            elif len(translated_text) == 2:
                character_of_interest = translated_text[1]
            # character_of_interest = translated_text[0]

            glyphs_tree = glyphTree(character_of_interest)
            # glyphs_tree.fill_tree()

            # translated_text = "好"
            # 怕 - phono-semantic
            # 好 - ideogrammic

            target_url = "https://en.wiktionary.org/wiki/" + urllib.parse.quote(character_of_interest)

            web_bytes = urllib.request.urlopen(target_url)

            if web_bytes != None:
                english_was_changed = False

                mybytes = web_bytes.read()

                wiki_html = mybytes.decode("utf8")
                web_bytes.close()

                soup = BeautifulSoup(wiki_html, 'html.parser')
                # relevant_entries = soup.find('span', id="Glyph_origin").next_element #.prettify()

                # test if there's a glyph origin entry
                glyph_origin = soup.find('span', id="Glyph_origin")
                if glyph_origin != None:
                    # figure out if phono-semantic or ideographic
                    # print(glyph_origin)
                    text_entry = glyph_origin.parent.find_next_sibling("p")

                    p_text = text_entry.getText()
                    words_of_text = p_text.split(" ")
                    compund_type = words_of_text[0]
                    print(compund_type)

                    all_hani_mentions = text_entry.find_all('i', {"class": "Hani mention"}) + text_entry.find_all(
                        'span', {"class": "Hani"})
                    print("number of hani entries: " + str(len(all_hani_mentions)))
                    for hani_mention in all_hani_mentions:
                        a_entry = hani_mention.findChildren("a")
                        chinese_char = a_entry[0].getText()
                        print(chinese_char)
                print("=====================")

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        # img_pil.paste((200, 200, 200), [0, 0, img.size[0], img.size[1]])
        # ImageDraw.rectangle([(100, 100), (120, 120)], fill=(255, 255, 255), outline=(0, 0, 0), width=4)

        the_text = "端午节就要到了。。。"
        the_text = translated_text
        draw.text((50, 80), the_text, font=font, fill=(b, g, r, a))
        img = np.array(img_pil)

        cv2.putText(img, english_text, (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1, cv2.LINE_AA)

        graph_im = glyphs_tree.render_tree()

        cv2.imshow("oth", graph_im)

        cv2.imshow("res", img)
        key = cv2.waitKeyEx(1)
        if key == 8:
            if len(english_text) <= 1:
                english_text = ""
            else:
                english_text = english_text[:-1]
                english_was_changed = True
        elif key == ord('q'):
            break
        elif not key == -1:
            english_text += (chr(key))
            if len(chr(key)) != 0:
                english_was_changed = True


if __name__ == "__main__":
    main()

# print(soup.prettify())
