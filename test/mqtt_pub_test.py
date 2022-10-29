import paho.mqtt.publish as publish

msgs = \
[
    {
        'topic':"/bm/test",
        'payload':"multiple 1"
    },

    (
        "/bm/test",
        "multiple 2", 0, False
    )
]
publish.multiple(msgs, hostname="test.mosquitto.org")
#Topic /bm/test 에 문자값 multiple 1, multiple 2를 발행.