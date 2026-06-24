function radar_processor()
    clear; clc;
    
    % 设置文件路径
    inputFile = 'D:\LEARNING\AY2025-2026T1\MiliwaveRadar\1.26Test\1.26雷达数据4_多车_detection.xlsx';  % 输入文件路径
    outputFile = 'D:\LEARNING\AY2025-2026T1\MiliwaveRadar\1.26Test\Target4.xlsx';       % 输出文件路径
    
    fprintf('开始处理\n');
 
    try
        [~, ~, rawData] = xlsread(inputFile);
        fprintf('原始数据读取完成\n', size(rawData, 1));
    catch
        error('无法读取文件，请检查文件路径和格式');
    end

    processedData = {};
    
    sectionStarts = [];
    for i = 1:size(rawData, 1)
        if iscell(rawData) && i <= size(rawData, 1) && ~isempty(rawData{i,1})
            if ischar(rawData{i,1}) && contains(string(rawData{i,1}), 'Timestamp_Seconds')
                sectionStarts = [sectionStarts, i];
            end
        end
    end
    
    fprintf('找到 %d 个子表格\n', length(sectionStarts));

    for secIdx = 1:length(sectionStarts)
        fprintf('正在处理子表格 %d/%d...\n', secIdx, length(sectionStarts));

        startRow = sectionStarts(secIdx);
        if secIdx < length(sectionStarts)
            endRow = sectionStarts(secIdx+1) - 1;
        else
            endRow = size(rawData, 1);
        end
        
        currentSection = rawData(startRow:endRow, :);
        
        % 提取时间戳
        timestampSec = [];
        timestampNano = [];
        dataStartRow = [];
        colHeaders = {};
        
        for i = 1:size(currentSection, 1)
            if iscell(currentSection) && i <= size(currentSection, 1) && ~isempty(currentSection{i,1})
                if ischar(currentSection{i,1})
                    cellContent = string(currentSection{i,1});
                    if contains(cellContent, 'Timestamp_Seconds')
                        if size(currentSection, 2) >= 2
                            timestampSec = currentSection{i,2};
                        end
                    elseif contains(cellContent, 'Timestamp_Nanoseconds')
                        if size(currentSection, 2) >= 2
                            timestampNano = currentSection{i,2};
                        end
                    elseif contains(cellContent, 'List_NumOfDetections')
                        dataStartRow = i + 2;  % 跳过列标题行
                        if i + 1 <= size(currentSection, 1)
                            colHeaders = currentSection(i + 1, :);
                        end
                        break;
                    end
                end
            end
        end
        
        if isempty(dataStartRow) || dataStartRow > size(currentSection, 1)
            fprintf('  子表格 %d 无数据部分，跳过\n', secIdx);
            continue;
        end
        
        % 获取列索引
        colIdx = zeros(1,5);
        for i = 1:length(colHeaders)
            if iscell(colHeaders) && i <= length(colHeaders) && ~isempty(colHeaders{i})
                if ischar(colHeaders{i})
                    headerText = string(colHeaders{i});
                    switch headerText
                        case 'x(m)', colIdx(1) = i;
                        case 'y(m)', colIdx(2) = i;
                        case 'z(m)', colIdx(3) = i;
                        case 'RangeRate(m/S)', colIdx(4) = i;
                        case 'RCS(dBm2)', colIdx(5) = i;
                    end
                end
            end
        end
        
        if any(colIdx == 0)
            fprintf('  子表格 %d 缺少必要的列，跳过\n', secIdx);
            continue;
        end
        
        % 处理数据行
        dataRows = currentSection(dataStartRow:end, :);
        validRows = 0;
        
        for j = 1:size(dataRows, 1)
            if iscell(dataRows) && j <= size(dataRows, 1)
                row = dataRows(j, :);
                
                % 检查行
                if isempty(row{1}) || (ischar(row{1}) && isempty(strtrim(row{1})))
                    continue;
                end
                
                % 提取数据
                if all([colIdx] <= length(row))
                    x = row{colIdx(1)}; 
                    y = row{colIdx(2)}; 
                    z = row{colIdx(3)};
                    rangeRate = row{colIdx(4)}; 
                    rcs = row{colIdx(5)};
                    
                    % 数据验证
                    if isnumeric(x) && isnumeric(y) && isnumeric(z) && ...
                       isnumeric(rangeRate) && isnumeric(rcs) && ...
                       abs(rangeRate) >= 1.5
                        
                        processedData = [processedData; {timestampSec, timestampNano, x, y, z, rangeRate, rcs}];
                        validRows = validRows + 1;
                    end
                end
            end
        end
        
        fprintf('  子表格 %d: 处理了 %d 行有效数据\n', secIdx, validRows);
    end
    
    % 保存结果
    if ~isempty(processedData)
        outputTable = cell2table(processedData, ...
            'VariableNames', {'Timestamp_Seconds', 'Timestamp_Nanoseconds', ...
                            'x_m', 'y_m', 'z_m', 'RangeRate_mS', 'RCS_dBm2'});

        writetable(outputTable, outputFile);
        fprintf('\n共 %d 行数据保存到: %s\n', size(processedData, 1), outputFile);
        
        % 显示结果预览
        fprintf('\n前5行结果:\n');
        disp(outputTable(1:min(5, height(outputTable)), :));
        
    else
        fprintf('未找到符合条件的数据\n');
    end
end

radar_processor();