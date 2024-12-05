


fkeys = get_range_CMIPC_data_fileKeys(year, day, hour, minute, bandNum)

if idx < len(fkeys):
    ds = retrieve_s3_data(fkeys[idx])

    show_CMPIC_image(ds) # << format shit
    ds.close()
