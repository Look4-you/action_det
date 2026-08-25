image_tag=vapd-recheck
tag=$1

if [ -z "$tag" ]; then
    echo "Usage:  <image_tag>"
    exit 1
fi

docker build -f ./Dockerfile -t $image_tag:$tag ..

imageId=$(docker images | grep "$image_tag.*$tag" | awk '{print $3}')
echo "imageId: $imageId"