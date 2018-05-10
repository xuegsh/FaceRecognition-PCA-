close all;
clear;

% ��ȡͼ��------------------------------------------------------------------
path = 'C:\Users\asus\Desktop\att_faces';
trainImages = [];
testImages = [];

files = dir(path);
index_for_test = randperm(10, 3);  % ����ÿ���˵�10��ͼ�����ѡ��7������ѵ��������3�����ڲ��ԡ�
for i = 4 : length(files)
    % personFile = dir([path, '\', files(i).name]);
    for j = 1 : 10
        img = imread([path, '\', files(i).name, '\', num2str(j), '.pgm']);
        [m, n] = size(img);
        img = double(reshape(img, m*n, 1));
        
        if ~ismember(j, index_for_test)
            trainImages = [trainImages, img];
        else
            testImages = [testImages, img];
        end
    end
end


meanImg = mean(trainImages, 2);
% display the mean image
figure(1); imshow(uint8(reshape(meanImg, 112, 92))); title('Mean Image');
X = trainImages - repmat(meanImg, 1, 7*40);

L = X' * X;
[V, D] = eig(L);  % ��L������ֵ����������

% ��Э�������C����������,�õ���������eigen_faces
eigen_faces = X * V;
% Unit-normalize the columns of 'eigen_faces'
eigen_faces = eigen_faces ./ (ones(size(eigen_faces,1),1) * sqrt(sum(eigen_faces .* eigen_faces)));
% display some of the 'eigen_faces'
figure(2);
for i = 1 : 8
    subplot(2,4,i);
    colormap('gray');
    imagesc(reshape(eigen_faces(:,i), 112, 92));
    title(['Eigenfaces ' num2str(i)]); 
end 


% ��������ά��k����ΧΪ50-100
for k = 50 : 100
    % �õ�����k��eigen_faces
    [D_sort, sort_index] = sort(sum(D), 'descend');
    sorted_eigen_faces = eigen_faces(:, sort_index);
    k_largest_eigen_faces = sorted_eigen_faces(:, 1:k); 


    % ��ÿ��mean-deducted��ѵ��ͼ��X(i)ͶӰ��������k���������������γɵ�eigenspace
    X_eigen_coefficients = [];
    for i = 1 : size(X, 2)
        X_eigen_coefficients(:, i) =  k_largest_eigen_faces' * X(:, i);
    end

    % ��ÿ��mean-deducted��testͼ��Y(i)ͶӰ��������k���������������γɵ�eigenspace
    Y = testImages - repmat(meanImg, 1, 3*40);
    Y_eigen_coefficients = [];
    for i = 1 : size(Y, 2)
        Y_eigen_coefficients(:, i) =  k_largest_eigen_faces' * Y(:, i);
    end

    % 2������Сƥ�䣬�õ�����ͼ��Y(i)��ƥ����
    % ��ŷ����þ���
    for i = 1 : size(Y, 2)
        minDist = Inf;
        minIndex = -1;
        for j = 1 : size(X, 2)
            distance(i, j) = norm(Y_eigen_coefficients(:,i) - X_eigen_coefficients(:, j), 2);
            if distance(i, j) < minDist
                minDist = distance(i, j);
                minIndex = j;
            end
        end
        match_result(i) = ceil(minIndex / 7);
    end


    % ������ȷ��
    correct_times(k) = 0;
    for i =  1 : size(Y, 2)
        if match_result(i) == ceil(i/3)
            correct_times(k) = correct_times(k)+1;
        end
    end
    correct_rate(k) = correct_times(k) / 120;
end