def analyse(item, with_d=False):
    print()
    print("Type:", type(item))
    print("Item:", item)

    try:
        print("Leng:", len(item))
    except Exception:
        print("function len() is not applicable to this object.")
    
    if with_d:
        print("Dimension:", item.shape)
    
    print()

