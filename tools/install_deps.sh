sudo apt install -y build-essential libtool autoconf automake libnuma-dev unzip pkg-config librdmacm-dev rdma-core

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../ &> /dev/null && pwd )"

rm -rf zeromq-4.1.4.tar.gz zeromq-4.1.4

wget https://raw.githubusercontent.com/mli/deps/master/build/zeromq-4.1.4.tar.gz
tar --no-same-owner -zxf zeromq-4.1.4.tar.gz
pushd zeromq-4.1.4 || exit
export CFLAGS=-fPIC
export CXXFLAGS=-fPIC

./configure -prefix=${THIS_DIR}/deps/ --with-libsodium=no --with-libgssapi_krb5=no
make -j
make install
popd || exit

rm -rf zeromq-4.1.4.tar.gz zeromq-4.1.4