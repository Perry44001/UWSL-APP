for i = 1: length(files_2)-1
    filepath = files_2(i);
    copyfile(filepath, pwd)
end
