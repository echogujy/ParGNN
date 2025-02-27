## create dir
mkdir -p ./log
sh ogbn-products.sh >> ./log/ogbn-products.txt
sh ogbn-proteins.sh >> ./log/ogbn-proteins.txt
sh yelp.sh >> ./log/yelp.txt
sh reddit.sh >> ./log/reddit.txt