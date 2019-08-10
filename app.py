import design_adviser

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators, SelectMultipleField
from wtforms.widgets import html_params
import pickle
import os
import numpy as np

app = Flask(__name__)


######## Preparing Classifier

def predict(text, input_values):
    x = np.array([design_adviser.vectorize(text, input_values)])
    attr_y = design_adviser.predict_attr(x)
    config_dict = design_adviser.config_dict
    attr_result = [config_dict['output'][i] for i, y in enumerate(attr_y) if y >= 0.5]
    color_y = np.round(design_adviser.predict_color(x) * 255).astype(int)
    reshaped_color_y = np.reshape(color_y, (5, 3))
    color_result = ['#%02x%02x%02x' % tuple(rgb_arr[::-1]) for rgb_arr in reshaped_color_y]
    return attr_result, color_result


######## Flask
def select_multi_checkbox(field, ul_class='', **kwargs):
    kwargs.setdefault('type', 'checkbox')
    field_id = kwargs.pop('id', field.id)
    html = [u'<ul %s>' % html_params(id=field_id, class_=ul_class)]
    for value, label, checked in field.iter_choices():
        choice_id = u'%s-%s' % (field_id, value)
        options = dict(kwargs, name=field.name, value=value, id=choice_id)
        if checked:
            options['checked'] = 'checked'
        html.append(u'<li><input %s /> ' % html_params(**options))
        html.append(u'<label for="%s">%s</label></li>' % (field_id, label))
    html.append(u'</ul>')
    return u''.join(html)

class ConceptForm(Form):
    category_choices = [
        [
            'Webサイト',
            'LP（ランディングページ）',
            'コンテンツページ'
        ],
        [
            'コーポレートサイト',
            'ブランドサイト･サービスサイト',
            'ECサイト･オンラインショップ',
            'キャンペーン･イベント･特設･プロモーションサイト',
            'ポータルサイト･メディア･マガジン･情報サイト',
            'プラットフォーム･コミュニティサイト',
            '採用サイト'
        ],
        [
            'コンピューター･Web･IT･AI･テクノロジー･通信関連',
            'サービス･アプリ･ツール',
            'プロジェクト',
            '音楽･ミュージック･芸能',
            '病院･クリニック･歯医者･医療',
            'ホテル･旅館',
            'カフェ･レストラン･飲食店･居酒屋･食品製造',
            '学校･教育･幼稚園･保育園･スクール',
            '科学･研究',
            '美容室･サロン･エステ･ヨガ',
            '旅行･観光･地域',
            '農業･農園･自然･酪農',
            'ペット･動物･生き物･植物',
            '制作会社･代理店･企画･マーケティング･ポートフォリオ',
            '体験･交流･遊び',
            '歴史･文化財･伝統',
            '求人･マッチング･転職',
            '建築･建設･不動産･家･庭',
            '生活･雑貨･おもちゃ･文具･家具',
            '家電･生活用品',
            '暮らし･インフラ･工業･メーカー',
            'デザイン･ものづくり･写真･動画･映像',
            'カルチャー･芸術',
            '介護･お年寄り',
            '寺･神社･葬儀',
            '車･バイク･乗り物･飛行機',
            '施設･スペース･レジャー施設･テーマパーク',
            '健康･運動･スポーツ･ジム',
            '美容･化粧品･コスメ･ケア用品',
            'ファッション･アクセサリー･ジュエリー',
            'ウェディング',
            'ベビー･子供･キッズ･ママ',
            'キャラクター',
            '漫画･アニメ･映画･ゲーム･テレビ･本',
            '料理･食べ物･飲み物',
            'チャリティー･NPO･エコ',
            '銀行･保険･金融･お金･法律'
        ]
    ]
    style_choices = [
        'シンプル',
        '高級感･リッチ･ゴージャス',
        'スタイリッシュ',
        '上品',
        'かわいい',
        'ポップ',
        'にぎやか',
        'かっこいい',
        'かため･かっちり',
        'やさしい･ナチュラル',
        'さわやか',
        '清潔感･信頼感',
        'グラデーション',
        'フラット･ベタ塗り',
        'ポリゴン',
        '迫力',
        'イラスト･アイコン',
        '漫画風･アメコミ風',
        '手書き感･コラージュ',
        '雑誌風',
        '質感･柄',
        '水彩･絵の具',
        '飾り罫',
        '和風',
        '花･木･葉･植物',
        '水',
        '文字',
        '形を多用',
        '季節感･季節のイベント',
        '春',
        '夏',
        '秋',
        '冬',
        '女性向け･女性的',
        '男性向け･男性的'
    ]
    color_choices = [
        'ブラック･黒色',
        'ホワイト･白色',
        'グレー･灰色',
        'ピンク･桃色',
        'レッド･赤色',
        'オレンジ･橙色',
        'イエロー･黄色',
        'ブルー･青色',
        'ネイビー･紺色',
        'グリーン･緑色',
        'ベージュ･肌色',
        'ブラウン･茶色',
        'パープル･紫色',
        'ゴールド･金色',
        'シルバー･銀色',
        'カラフル･多色',
        '暖色',
        '寒色',
        '中性色',
        '白黒･モノトーンな配色',
        'あわい･パステルな配色',
        'やさしい･やわらかい配色',
        '元気･楽しい･にぎやかな配色',
        '健康的･ヘルシー･リラックス･フレッシュな配色',
        'かわいい配色',
        '高級感がある配色',
        '強い･あざやか･ポップな配色',
        'ダーク･重厚感のある配色',
        '落ち着いた配色',
        'スタイリッシュ･知的な配色',
        'さわやかな配色',
        '和･レトロな配色',
        'キーカラーが効いてる配色',
        '組み合わせが効いてる配色',
        '奇抜･めずらしい･派手な配色'
    ]
    category1 = SelectMultipleField(
        '', choices=[(val, val) for val in category_choices[0]], widget=select_multi_checkbox)
    category2 = SelectMultipleField(
        '', choices=[(val, val) for val in category_choices[1]], widget=select_multi_checkbox)
    category3 = SelectMultipleField(
        '', choices=[(val, val) for val in category_choices[2]], widget=select_multi_checkbox)
    style = SelectMultipleField(
        '', choices=[(val, val) for val in style_choices], widget=select_multi_checkbox)
    color = SelectMultipleField(
        '', choices=[(val, val) for val in color_choices], widget=select_multi_checkbox)
    description = TextAreaField('Description',
        [])

@app.route('/')
def index():
    form = ConceptForm(request.form)
    return render_template('form.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ConceptForm(request.form)
    if request.method == 'POST':
        text = request.form['description']
        category1 = request.form.getlist('category1')
        category2 = request.form.getlist('category2')
        category3 = request.form.getlist('category3')
        style = request.form.getlist('style')
        color = request.form.getlist('color')
        print(category1, category2, category3, style, color)
        input_values = category1 + category2 + category3 + style + color
        attr_result, color_result = predict(text, input_values)
        return render_template('results.html',
            input_values=input_values,
            description=text,
            attr=attr_result,
            color=color_result)
    return render_template('reviewform.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
