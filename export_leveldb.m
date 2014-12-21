function export_leveldb(X, y, height, width, channels, label)
%EXPORT_LEVELDB Summary of this function goes here
%   Detailed explanation goes here

addpath('/Users/Calvin/Github/matlab-leveldb');
addpath('/Users/Calvin/Github/protobuf-matlab/protobuflib/');
addpath('protobuf');

db_path = sprintf('caffe/%s_ldb', label);
assert(height * width * channels == size(X,2));

do_export_leveldb(db_path, X, y, channels, height, width);

function do_export_leveldb( db_path, X, y, channels, height, width)
%DO_EXPORT_LEVELDB 
%   It does.

n = size(X, 1);

%% Export to the LevelDB.
database = leveldb.DB(db_path);
batch = leveldb.WriteBatch();

for i = 1 : n
    bob = pb_read_Datum();
    bob = pblib_set(bob, 'channels', channels);
    bob = pblib_set(bob, 'height', height);
    bob = pblib_set(bob, 'width', width);
    bob = pblib_set(bob, 'data', uint8(X(i,:)*255));
    bob = pblib_set(bob, 'label', y(i));
    
    key = sprintf('%08d', i);
    value = pblib_generic_serialize_to_string(bob);
    batch.put(key, value);
    
    if 0 == mod(i, 1000)
        database.write(batch);
        batch = leveldb.WriteBatch();
    end
end

% write the last batch
if 0 ~= mod(i, 1000)
    database.write(batch);
end
clear database;
