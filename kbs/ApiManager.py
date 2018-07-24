import sys
from enum import Enum

import requests
from typing import List
from kbs.models import *

class ApiManager(object):
    """
    The API Manager. It simply implements the interface for interact with the KB
    """

    class Method(Enum):
        GET = 1
        POST = 2

    def __init__(self, key, prot, host, port, path):
        self.key = key
        self.prot  =prot
        self.host = host
        self.port = port
        self.path = path
        self.baseurl = prot+"://"+host + ":" + port + path

    @classmethod
    def init_from_conf(cls, config):
        return cls(
            config["key"],
            config["prot"],
            config["host"],
            config["port"],
            config["path"],
        )


    def items_number_from(self, id) -> int:
        method_name = self.items_number_from.__name__
        response = self._call(method_name, {"id":id, "key":self.key},method=ApiManager.Method.GET)
        return int(response.text)

    def items_from(self, id) -> List[Item]:
        method_name = self.items_from.__name__
        response = self._call(method_name, {"id":id, "key":self.key}, method=ApiManager.Method.GET)
        json = response.json()
        result = []
        for obj in json:
            item = Item.init_from_json(obj)
            result.append(item)

        return result

    def _remove_item_fields(self, item_json):
        if "_id" in item_json:
            del item_json["_id"]
        if "HASH" in item_json:
            del item_json["HASH"]
        return item_json


    def add_item(self, item:Item, is_test=True) -> None:
        method_name = self.add_item.__name__
        if is_test: method_name+="_test"
        item_json = item.to_json()
        item_json = self._remove_item_fields(item_json)
        response = self._call(method_name, {"key": self.key}, json=item_json, method=ApiManager.Method.POST)
        json = response.json()
        # it should be 1
        return json


    def add_items(self, items:List[Item], is_test=True):
        method_name = self.add_items.__name__
        if is_test: method_name += "_test"
        items_json = []
        for it in items:
            item_json = self._remove_item_fields(it.to_json())
            items_json.append(item_json)
        response = self._call(method_name, {"key": self.key}, json=items_json, method=ApiManager.Method.POST)
        json = response.json()
        # it should be 1
        return json


    def _call(self, method_name, params, method=Method.GET, json=None):
        req_url = self.baseurl + method_name
        if method==ApiManager.Method.GET:
            return requests.get(req_url, params=params, )
        else:
            return requests.post(req_url, params=params, json=json)

