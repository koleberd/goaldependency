#used to remove an object from a list by object
def remove(ls,obj):
    for key, val in enumerate(ls):
        if id(val) == id(obj):
            del ls[key]
            break
