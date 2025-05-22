def recursive_list_cast(obj):
    if isinstance(obj, range):
        return list(obj)
    
    elif isinstance(obj, dict):
        return {k: recursive_list_cast(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [recursive_list_cast(item) for item in obj]
    return obj