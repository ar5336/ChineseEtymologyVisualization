
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


IMAGE_WIDTH = 900
IMAGE_HEIGHT = 600


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class glyphNode:
    def __init__(self, glyph, offset, parent):
        self.grid_position = self.get_position(offset, parent)
        self.offset = offset
        self.glyph = glyph
        self.pinyin = pinyin.get(glyph, format="numerical")
        english = pinyin.cedict.translate_word(glyph)
        if english is not None:
            # print(english[0])
            self.english = english[0]
        else:
            self.english = ""
        self.is_explored = False

        self.parent_node = parent
        self.children_nodes = []
        self.children_type = ""

    def get_position(self, offset, parent):
        if parent is None:
            return Point(offset.x, offset.y)
        return Point(parent.grid_position.x + offset.x, parent.grid_position.y + offset.y)

    def recalc_position(self):
        if self.parent_node is not None:
            self.grid_position = Point(self.parent_node.grid_position.x + self.offset.x, self.parent_node.grid_position.y + self.offset.y)

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

        rect_x1 = int((self.grid_position.x) * render_scale + IMAGE_WIDTH / 2)
        rect_y1 = int(((self.grid_position.y) * render_scale) + IMAGE_WIDTH / 10)
        # if self.parent_node is not None:
        #     rect_x1 = int((self.parent_node.grid_position.x + self.grid_position.x) * render_scale + IMAGE_WIDTH / 2)
        #     rect_y1 = int(((self.parent_node.grid_position.y + self.grid_position.y) * render_scale) + IMAGE_WIDTH / 10)
        rect_x2 = int(rect_x1 + render_scale * block_size)
        rect_y2 = int(rect_y1 + render_scale * block_size)

        rected_im = cv2.rectangle(image, [rect_x1, rect_y1], [rect_x2, rect_y2], (255,255,255), thickness=-1)

        derivation_label = ""
        derivation_label = self.children_type

        # draw branches to children
        if self.children_nodes is not None:
            for child_index in range(len(self.children_nodes)):
                individual_label = ""

                if self.children_type == "Phono-Semantic_Compound":
                    if child_index == 0:
                        individual_label = "Semantic"
                    elif child_index == 1:
                        individual_label = "Phonetic"

                child_node = self.children_nodes[child_index]
                child_x_1 = int(child_node.grid_position.x * render_scale + IMAGE_WIDTH / 2)
                child_y_1 = int(child_node.grid_position.y * render_scale + IMAGE_WIDTH / 10)

                # if self.parent_node is not None:
                #     child_x_1 = int(child_node.grid_position.x * render_scale + IMAGE_WIDTH / 2)
                #     child_y_1 = int(child_node.grid_position.y * render_scale + IMAGE_WIDTH / 10)
                # individual label
                rected_im = cv2.putText(rected_im, individual_label, (child_x_1, child_y_1-20), cv2.FONT_HERSHEY_PLAIN, .7, (255,255,255), 1, cv2.LINE_AA)


                rected_im = cv2.line(rected_im, (rect_x1 + int(render_scale/2), rect_y2), (child_x_1 + int(render_scale/2),  child_y_1), (255,255,255), thickness=3)

        # derivation label
        rected_im = cv2.putText(rected_im, derivation_label, (rect_x1 - 20, rect_y1 + render_scale + 23),
                                cv2.FONT_HERSHEY_PLAIN, .8, (50, 50, 225), 1, cv2.LINE_AA)

        cv2.putText(rected_im, self.english, (rect_x1 + render_scale + 4, int(rect_y1 + render_scale/2)), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(rected_im, self.pinyin, (rect_x1 + render_scale + 4, int(rect_y1 + render_scale)),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, "--- by Silencer", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1, cv2.LINE_AA)

        img_pil = Image.fromarray(rected_im)
        draw = ImageDraw.Draw(img_pil)

        fontpath = "./simsun.ttc"
        font = ImageFont.truetype(fontpath, render_scale-4)

        b, g, r = 0, 0, 0
        draw.text((int(rect_x1), int(rect_y1)), self.glyph, font=font, fill=(b,g,r))

        img = np.array(img_pil)

        return img

    def scan_children(self):
        derivation_type, child_glyphs = get_components_from_glyph(self.glyph)
        self.children_type = derivation_type
        # child_index = 0
        num_children = len(child_glyphs)
        for child_index in range(num_children):
            if num_children == 1:
                self.children_nodes.append(glyphNode(child_glyphs[child_index],Point(0,1.5), self))
            elif num_children == 2:
                self.children_nodes.append(glyphNode(child_glyphs[child_index],Point( -2 + 4 * child_index,1.5), self))
            else:
                print("too many children!")

    def recalc_child_pos(self):
        num_children = len(self.children_nodes)
        if num_children == 1:
            self.children_nodes[0].recalc_position()
        for child_index in range(len(self.children_nodes)):
            child = self.children_nodes[child_index]

            child.recalc_position()
            child.recalc_child_pos()

def get_components_from_glyph(glyph):
    # scrape html
    # return composition type and glyphs with optional derivation types

    target_url = "https://en.wiktionary.org/wiki/" + urllib.parse.quote(glyph)

    # web_bytes = urllib.request.urlopen(target_url)
    web_bytes = None
    try:
        web_bytes = urllib.request.urlopen(target_url)
        # break
    except urllib.error.HTTPError as exception:

        print("Wiktionary entry not found")
        return "", []

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
        if glyph_origin is not None:
            # figure out if phono-semantic or ideographic
            # print(glyph_origin)
            text_entry = glyph_origin.parent.find_next_sibling("p")

            p_text = text_entry.getText()
            words_of_text = p_text.split(" ")
            compund_type = words_of_text[0]
            print(compund_type)
            # derivation_type = compund_type

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

            if len(hani_glyphs) > 0:
                if compund_type == "Variant":
                    derivative_glyphs = [hani_glyphs[0]]
                    derivation_type = "Variant"
                elif compund_type == "Simplified":
                    # print(hani_glyphs)
                    derivative_glyphs = [hani_glyphs[0]]
                    derivation_type = "Simplified"
                elif hani_glyphs[0] == "形聲": #phonosemantic
                    phonetic_index = 0
                    if "phonetic" in words_of_text: #words_of_text.contains("phonetic"):
                        phonetic_index = words_of_text.index("phonetic")
                    semantic_index = 1
                    if "semantic" in words_of_text: #words_of_text.contains("semantic"):
                        semantic_index = words_of_text.index("semantic")
                    if phonetic_index < semantic_index:
                        derivative_glyphs = [hani_glyphs[2], hani_glyphs[1]]
                    else:
                        derivative_glyphs = hani_glyphs[1:3]
                    derivation_type = "Phono-Semantic_Compound"
                elif hani_glyphs[0] == "象形": #pictogram
                    derivation_type = "Pictogram"
                    if len(hani_glyphs) >= 2:
                        if hani_glyphs[1] == "女":
                            derivative_glyphs = []
                        else:
                            derivative_glyphs = []
                            # derivative_glyphs = hani_glyphs[1:3]
                        # derivative_glyphs = hani_glyphs[1::]
                    else:
                        derivative_glyphs = []
                elif hani_glyphs[0] == "會意": #ideogrammic
                    derivative_glyphs = hani_glyphs[1:3]
                    derivation_type = "Idogrammic_Compound"
                    # print(derivative_glyphs)
                else:
                    print("unrecognized glyph origin entry")
        else:
            # no 'glyph origin' entry

            # grab the thing after the 'chinese' entry
            chinese_entry = soup.find('span', id="Chinese")

            simp_table = None
            if chinese_entry is not None:
                simp_table = chinese_entry.parent.find_next("table")

            if simp_table is not None:
                table_text = simp_table.getText()
                glyph_index = -1
                table_words = table_text.split(" ")
                for word_index in range(len(table_words)):
                    word = table_words[word_index]
                    if word == glyph:
                        glyph_index = word_index
                        break
                trad_char = ""
                if glyph_index != -1:
                    trad_char = table_words[glyph_index + 3]
                    derivative_glyphs = [trad_char]
                    derivation_type = "Simplified"

            # however there is a leftover possibility that the "simplified" infromation is in the table
        # print(relevant_entries)
        return derivation_type, derivative_glyphs
    return "", []


class glyphTree:
    def __init__(self, starting_glyph):
        self.starting_glyph = starting_glyph
        self.root_node = glyphNode(starting_glyph, Point(0, 0), None)

        # init stuff
        self.fill_tree()

    def fill_tree(self):
        keep_scanning = True
        # current_node = self.root_node
        nodes_to_explore = [self.root_node]
        while len(nodes_to_explore) != 0:
            current_node = nodes_to_explore.pop()
            # scan wiki for component parts
            # this is where chaching will go
            current_node.scan_children()

            num_children = len(current_node.children_nodes)
            print("the number of children is: "+str(num_children))
            for child_index in range(num_children):
                nodes_to_explore.append(current_node.children_nodes[child_index])

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
        graph_canvas = np.zeros((700, 900, 3), np.uint8)
        # graph_canvas = np.zeros((500, 500, 3))

        tree_list = self.tree_to_list()
        for glyph_node in tree_list:
            graph_canvas = glyph_node.render_node(graph_canvas)
        return graph_canvas

    def check_for_overlap(self):
        tree_list = self.tree_to_list()
        num_nodes = len(tree_list)
        overlaps = 0
        node_pairs = []
        if num_nodes > 3:
            for node_1_index in range(num_nodes-1):
                node_1 = tree_list[node_1_index]
                for node_2_index in range(node_1_index+1, num_nodes):
                    node_2 = tree_list[node_2_index]
                    p1 = node_1.grid_position
                    p2 = node_2.grid_position

                    if p1.x == p2.x and p1.y == p2.y:
                        overlaps += 1
                        node_pairs.append([node_1,node_2])
        print("there are "+str(overlaps)+" overlap")

        for overlap_index in range(overlaps):
            n1, n2 = node_pairs[overlap_index]

            offset = 1.5

            #move parents farther apart
            p1 = n1.parent_node
            p2 = n2.parent_node
            if p1.grid_position.x < p2.grid_position.x:
                # if parent 1 is to the left of parent 2
                p1pos = p1.grid_position
                p2pos = p2.grid_position

                # move parent 1 left
                p1.offset = Point(p1.offset.x-offset, p1.offset.y)
                # p1.grid_position = Point(p1pos.x-2, p1pos.y)
                # move parent 2 right
                p2.offset = Point(p2.offset.x+offset, p2.offset.y)
                # p2.grid_position = Point(p2pos.x+2, p1pos.y)
            else:
                # if parent 2 is to the left of parent 1
                p1pos = p1.grid_position
                p2pos = p2.grid_position

                # move parent 1 left
                p1.offset = Point(p1.offset.x + offset, p1.offset.y)
                # p1.grid_position = Point(p1pos.x + 2, p1pos.y)
                # move parent 2 right
                p2.offset = Point(p2.offset.x - offset, p2.offset.y)
                # p2.grid_position = Point(p2pos.x - 2, p1pos.y)

def main():
    selected_char_index = 0

    # fontpath = "./simsun.ttc" # <== 这里是宋体路径
    fontpath = "./simsun.ttc"
    font = ImageFont.truetype(fontpath, 32)

    # cv2.putText(img,  "--- by Silencer", (200,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv2.LINE_AA)
    translator = Translator()

    english_text = "steel"

    # stationary = 静止的
    # campsite


    # english_was_changed = True
    #
    # should_translate = True
    do_translation = True
    do_render = True

    glyphs_tree = None
    selected_character = ""

    translated_text = ""
    while True:
        img = np.zeros((600, 400, 3), np.uint8)

        b, g, r, a = 255, 255, 255, 255

        # if should_translate:
        # if english_was_changed or should_translate:

        if do_translation:
            translation = ""
            if len(english_text) >= 1:
                translation = translator.translate(english_text, src='en', dest='zh-CN')

            translated_text = translation.text

            # translated_text = "好"
            # 怕 - phono-semantic
            # 好 - ideogrammic
            do_translation = False

        if do_render:
            selected_character = translated_text[min(len(translated_text)-1, selected_char_index)]

            glyphs_tree = glyphTree(selected_character)

            glyphs_tree.check_for_overlap()
            glyphs_tree.root_node.recalc_child_pos()
            graph_im = glyphs_tree.render_tree()
            cv2.imshow("oth", graph_im)

            english_was_changed = False
            do_render = False

        the_text = "端午节就要到了。。。"
        # only do these if the english was changed

        should_translate = False

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        the_text = translated_text
        # write chinese font
        draw.text((50, 80), the_text, font=font, fill=(b, g, r, a))
        img = np.array(img_pil)

        offset = selected_char_index * 32

        rect_w = 32
        rect_h = 32
        cv2.rectangle(img, (50+offset, 80), (50+rect_w+offset, 80+rect_h), (200,125,125), thickness=2)
        cv2.putText(img, english_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1, cv2.LINE_AA)

        cv2.imshow("res", img)
        key = cv2.waitKeyEx(1)

        # up = 2490368
        # down = 2621440
        # left = 2424832
        # right = 2555904

        # if key != -1:
        #     print(key)
        if key == 2424832: # left
            selected_char_index = max(selected_char_index-1, 0)
            english_was_changed = True
            do_render = True
        elif key == 2555904: # right
            selected_char_index += 1
            english_was_changed = True
            do_render = True
        elif key == 13: #enter
            should_translate = True
            do_translation = True
            english_was_changed = True
            do_render = True
        elif key == 8: #backspace
            if len(english_text) <= 1:
                english_text = ""
            else:
                english_text = english_text[:-1]
                # english_was_changed = True
        elif key == ord('q'): #q to quit
            break
        elif not key == -1 and key <= 110000: #any other key
            english_text += (chr(key))
            if len(chr(key)) != 0:
                english_was_changed = True


if __name__ == "__main__":
    main()

# print(soup.prettify())
