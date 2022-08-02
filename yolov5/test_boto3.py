    import boto3
    s3 = boto3.resource('s3')
    ## Bucket to use
    bucket = s3.Bucket('equitable-surveillance-processed-output')

    imgs = []
    
    # bucket = s3.Bucket('bucket-name')
    objs = list(bucket.objects.filter(Prefix= '{}/crops/0/'.format('accident_scene_Trim_Trim')   ))
    for i in range(0, len(objs)):
        # print(objs[i].key)
        # imgs.append( 'https://equitable-surveillance-processed-output.s3.amazonaws.com/'.format(objs[i].key)  )
        imgs.append( 'https://equitable-surveillance-processed-output.s3.amazonaws.com/' + objs[i].key) 

    print(imgs)