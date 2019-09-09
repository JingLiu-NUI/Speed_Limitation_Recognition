function[] =mini_pro()
    % circle detection, using imfindcircles to detect the circle and get
    % the center point and radius, then set the attribute as bright and
    % objectPoarity will improve the accuracy of circle detection, through detect the
    % center point to judge whether its empty.
    function result = circle_detection(image)
        [centers, radius] = imfindcircles(image,[20 60],'ObjectPolarity','bright');
        %viscircles(centers, radius,'Color','b');
        result = ~isempty(centers);
    end
    % Used to detect red pixels in an image, its for H channel and S
    % channel, if the value greater than or less than this threahold,
    %then it will be judge as red pixel.
    function outresult = red_detection(imageRGB)
            % Convert RGB image to HSV
            imageHSV = rgb2hsv(imageRGB);
            % Threshold image
            outresult = (imageHSV(:, :, 1) >= 0.9 | imageHSV(:, :, 1) <= 0.1) &...
                (imageHSV(:, :, 2) >=  0.5 & imageHSV(:, :, 2) <= 1.0);
    end
    %this function is for caculate distance of two images, after image
    %processing, the image should be caculate the distance by two way,euclidean_dist
    %and Consine distance, the get the minimal one with gold standard image
    %and return the index of that image
    %reference: https://uk.mathworks.com/matlabcentral/answers/343645-how-to-find-the-euclidean-distance-between-two-images
    function speedLimit=distance_classify(digitROI,HOGfeature)
             % Load digits extracted from the gold standard images
            myFolders='C:\Users\Owner\Desktop\mini-pro\result2';
            filePattern=fullfile(myFolders,'*.jpg');
            jpgFiles=dir(filePattern);
            for k = 1:length(jpgFiles)
                baseFileName=jpgFiles(k).name;
                filePath=fullfile(myFolders,baseFileName);
                goldImage = imread(filePath);
                if strcmp(HOGfeature,'true')
                    goldImage=HOG(goldImage);
                end
                % Calculate the distance of euclidean
                euclidean_dist(k) = sqrt(sum((im2double(digitROI(:))- im2double(goldImage(:))).^ 2));
                %cosine_dis(k) = pdist2(digitROI(:),goldImage(:),'cosine');
            end
            % The output image with the highest percentage of black pixels is the
            % best match
            [~, index] = min(euclidean_dist);
            % Return string of the best matched speed
            
            speedLimit = jpgFiles(index).name(1:end-4);
        end
    
    %reference : https://uk.mathworks.com/matlabcentral/fileexchange/28689-hog-descriptor-for-matlab
    %this is HOG function, which used for extract the features, here I use
    %two way, fistly feed the image matrix after preprocessing, other way
    %is get the  feature vectors of each image and then feed into network
    %or distance
    function H=HOG(Im)
        nwin_x=3;%set here the number of HOG windows per bound box
        nwin_y=3;
        B=9;%set here the number of histogram bins
        [L,C]=size(Im); % L num of lines ; C num of columns
        H=zeros(nwin_x*nwin_y*B,1); % column vector with zeros
        m=sqrt(L/2);
        if C==1 % if num of columns==1
            Im=im_recover(Im,m,2*m);%verify the size of image, e.g. 25x50
            L=2*m;
            C=m;
            end
        Im=double(Im);
        step_x=floor(C/(nwin_x+1));
        step_y=floor(L/(nwin_y+1));
        cont=0;
        hx = [-1,0,1];
        hy = -hx';
        grad_xr = imfilter(double(Im),hx);
        grad_yu = imfilter(double(Im),hy);
        angles=atan2(grad_yu,grad_xr);
        magnit=((grad_yu.^2)+(grad_xr.^2)).^.5;
        for n=0:nwin_y-1
            for m=0:nwin_x-1
                cont=cont+1;
                angles2=angles(n*step_y+1:(n+2)*step_y,m*step_x+1:(m+2)*step_x); 
                magnit2=magnit(n*step_y+1:(n+2)*step_y,m*step_x+1:(m+2)*step_x);
                v_angles=angles2(:);    
                v_magnit=magnit2(:);
                K=max(size(v_angles));
                %assembling the histogram with 9 bins (range of 20 degrees per bin)
                bin=0;
                H2=zeros(B,1);
                for ang_lim=-pi+2*pi/B:2*pi/B:pi
                    bin=bin+1;
                    for k=1:K
                        if v_angles(k)<ang_lim
                            v_angles(k)=100;
                            H2(bin)=H2(bin)+v_magnit(k);
                        end
                    end
                end         
                H2=H2/(norm(H2)+0.01);        
                H((cont-1)*B+1:cont*B,1)=H2;
            end
         end
    end

    %this function is for processing image, firstly read the image from
    %file, then adjust the bright level of each image using RGB and HSV, 
    %and then detect the circle or red area to adjust whether it is a speed
    %limit sign, then according get image after binarize to remove some
    %small area or object which is white incase of keep the circle more
    %clean, then use the regionprops to automatically get the bounding of
    %circle box and then caculating the distance of the image after box
    %bounding, other way is coverting the image after box bounding to HOS
    %vector and feed it to calculate the diatance, the accuray some of type image like
    %20Kph improved while some of like 50Kph are reduced.
    function final_acc=Image_processing(myFolder,HOGfeature)
            %myFolder='C:\Users\Owner\Desktop\mini-pro\GoldStandards';
            filePattern=fullfile(myFolder,'*.jpg');
            jpgFiles=dir(filePattern);
            acc=0;
            count=0;
            final_edu_acc=0;
            final_acc=0;
            for g = 1:length(jpgFiles)
                baseFileName=jpgFiles(g).name;
                fullFileName=fullfile(myFolder,baseFileName);
                A=imread(fullFileName);
                A = imresize(A,[140 140]); 
               [m,n,k] = size(A);
                hsv = rgb2hsv(A);
                V = hsv(:,:,3); 
                if sum(V(:))<11000
                %1.7062e+05
                for i = 1:m 
                for j = 1:n 
                hsv(i,j,3) =2.0* hsv(i,j,3);
                end 
                end
                end
                S = hsv2rgb(hsv);
                gr=rgb2gray(S);
                B1=imbinarize(gr);             
                %figure,imshow(B1);
                result = circle_detection(B1);
                if result==1
                    acc=acc+1;
                    redBW = red_detection(A);
                    redBW = bwareaopen(redBW, 50);
                    redBW = imclearborder(redBW); 
                    se = strel('disk', 5);
                    redBW = imdilate(redBW, se);
                    redBW = imerode(redBW, se);
                    [m,n] = size(B1);
                    if n >= m   
                       image_extract = B1(1:m, 1:m);
                    else  
                       image_extract = B1(m-n:m-1, 1:n);
                    end
                    exreact_number = imresize(image_extract, [450, 450]);
                    c = bwareaopen(exreact_number, 50);
                    c = imclearborder(c);
                    se = strel('disk', 5);
                    % imerode and imdilate work better 
                    c = imerode(c, se);
                    c = imdilate(c, se);
                    % get all connect component
                    cc = bwconncomp(c);
                    lm = labelmatrix(cc);
                    le= regionprops(cc, 'Extent');
                    indexe = [le.Extent] >= 0.3;
                    mask = ismember(lm, find(indexe));
                    %extracts all connected components (objects) from the binary image BW
                    mask=bwareafilt(mask,2);   
                    statsBB = regionprops(mask, 'BoundingBox');
                    if ~isempty(statsBB)
                       count=count+1;
                       % Extract the ROI for the left digit by setting the index of the object
                       % in the labelled image to '1'
                       bbox = statsBB(1).BoundingBox;
                       digitROI = mask(int16(bbox(2)):int16(bbox(2)+bbox(4)), int16(bbox(1)):int16(bbox(1)+bbox(3)),:);
                       % Resize each image to [160, 120]
                       digitROI = imresize(digitROI, [160, 120]);   
                       %figure,imshow(digitROI); 
                       if strcmp(HOGfeature,'true')
                            digitROI=HOG(digitROI);
                       end
                       speedLimit = distance_classify(digitROI,HOGfeature);
                       if strcmp(myFolder,'C:\Users\Owner\Desktop\mini-pro\20Kph')
                           %download the image dataset 
                           %imwrite(digitROI, sprintf('./total2/20/%02d.jpg',g))
                            if strcmp(speedLimit,'02')
                                 final_edu_acc=final_edu_acc+1;
                            end
                       end
                       if strcmp(myFolder,'C:\Users\Owner\Desktop\mini-pro\30Kph')
                           %imwrite(digitROI, sprintf('./total2/40/%02d.jpg',g))
                            if strcmp(speedLimit,'03')
                                 final_edu_acc=final_edu_acc+1;
                            end
                       end    
                       if strcmp(myFolder,'C:\Users\Owner\Desktop\mini-pro\50Kph')
                            %imwrite(digitROI, sprintf('./total2/50/%02d.jpg',g))
                           if strcmp(speedLimit,'04') 
                               final_edu_acc=final_edu_acc+1;
                           end
                       end
                       if strcmp(myFolder,'C:\Users\Owner\Desktop\mini-pro\80Kph')
                           %imwrite(digitROI, sprintf('./total2/80/%02d.jpg',g))
                           if strcmp(speedLimit,'08')
                               final_edu_acc=final_edu_acc+1;
                           end
                       end
                       if strcmp(myFolder,'C:\Users\Owner\Desktop\mini-pro\100Kph')
                            %imwrite(digitROI, sprintf('./total2/100/%02d.jpg',g))
                           if strcmp(speedLimit,'010')
                                 final_edu_acc=final_edu_acc+1;
                           end
                       end
                    end
                 end
                end  
            final_acc=final_edu_acc/length(jpgFiles);
    end
    %using imagedatastore to read data and then using splitEachLabel to
    %split to taining dataset and test dataset, convert the iamge to HOG
    %feature vector, then using function fitcecoc svm classifier to fit the number and
    %then predict the testing the test dataset and get the accuracy of
    %testing dataset, the result is 0.7890
    function SVM_classify()
        imds = imageDatastore('C:\Users\Owner\Desktop\mini-pro\total2', 'IncludeSubfolders', true, 'labelsource', 'foldernames') ;        
        tbl = countEachLabel(imds);
        [train,test] = splitEachLabel(imds, 0.8,'randomize'); 
        %trainingFeatures = extract_feature(train)
        trainLebal=train.Labels;
        train_num= numel(train.Files);
        vectors_train= zeros(train_num, 81);
        for i=1:train_num
            image = readimage(train,i);      
            vectors_train(i,:)=HOG(image);
        end
        classifier = fitcecoc(vectors_train,trainLebal);
        testLabels=test.Labels;
        test_num=numel(test.Files);
        vectors_test=zeros(test_num, 81);
        
         for i=1:test_num
            image = readimage(test,i);
            vectors_test(i,:)=HOG(image);
        end
        predictedLabels =  predict(classifier,vectors_test);
        accuracy = sum(predictedLabels == testLabels)/numel(testLabels);
        disp('the accuracy for speed limit sign detection with SVM: ')
        disp(accuracy)
    end
    % for the street detection
    function street_detection()
        imds = imageDatastore('C:\Users\Owner\Desktop\mini-pro\stress dataset', 'IncludeSubfolders', true, 'labelsource', 'foldernames') ;
        label={'01','01','01','03','02','02','02','01','01','01','01','01','01','01','01','02'};
        street_acc=0;
        canbe_re=0;
        prediction={};     
        %get the image dataset
        for s = 1:numel(imds.Files)
            %read each image
            image = readimage(imds,s);
            %if the image size is not [480   640     3],then can not detect
            if(size(image)==[480   640     3])
                imagelight_resize = imresize(image,[450 450]);
                img_gray= rgb2gray(image);
                %pre-processing with image, convert it to binarize image and
                %resize it
                x =  imsubtract(image(:,:,1), img_gray);
                x = medfilt2(x, [3,3]);
                x = imbinarize(x, 0.18);
                x = bwareaopen(x, 300);    
                x = imresize(x, [450 450]);
                %detect the bounding box with regionprops function
                cc = bwconncomp(x);
                    stats = regionprops(cc, 'BoundingBox');
                if size(stats) ~=[0 0]
                    %if there is a circle exists
                     bbox = stats.BoundingBox;
                     %get the digit image
                     digitROI = imagelight_resize(int16(bbox(2)):min(int16(bbox(2)+bbox(4)), 450),... 
                     int16(bbox(1)):min(int16(bbox(1)+bbox(3)), 450), :);
                     image_street= imbinarize(rgb2gray(digitROI));    
                     number_street = imresize(image_street, [160, 120]);
                     %convet binary image to HOG features
                     number_street=HOG(number_street);
                     %calculate the distance between HOG feature of image
                     %and gold standard image
                     spped_li = distance_classify(number_street,'true');
                     canbe_re=canbe_re+1;
                else 
                     spped_li='200';
                end
            else
                spped_li='200';
            end
            prediction{end+1}=spped_li;
        end
        for i =1:length(prediction)
            if strcmp(prediction(i),label(i))
                street_acc=street_acc+1;
            end
        end
        %calculate the accuracy
        acc_final=street_acc/canbe_re;
        disp('the accuracy for street speed limit sign detection: ')
        disp(acc_final)
    end

    %Function for get each class image
    function speed_limit_detection()
        myFolder1='C:\Users\Owner\Desktop\mini-pro\20Kph';
        myFolder2='C:\Users\Owner\Desktop\mini-pro\30Kph';
        myFolder3='C:\Users\Owner\Desktop\mini-pro\50Kph';
        accuracy_20=Image_processing(myFolder1,'flase');
        accuracy_30=Image_processing(myFolder2,'flase');
        accuracy_50=Image_processing(myFolder3,'flase');
        accuracy_20_H=Image_processing(myFolder1,'true');
        accuracy_30_H=Image_processing(myFolder2,'true');
        accuracy_50_H=Image_processing(myFolder3,'true');
        disp('accuracy for 20Kph without feature extraction:');
        disp(accuracy_20);
        disp('accuracy for 20Kph with feature extraction:');
        disp(accuracy_20_H)
        disp('accuracy for 30Kph without feature extraction: ');
        disp(accuracy_30);
        disp('accuracy for 30Kph with feature extraction: ');
        disp(accuracy_30_H);
        disp('accuracy for 50Kph without feature extraction: ');
        disp(accuracy_50);
        disp('accuracy for 50Kph with feature extraction: ');
        disp(accuracy_50_H); 
    end
%you can invoke the function here
%street_detection()
%SVM_classify()
%speed_limit_detection()
end
