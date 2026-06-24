function RadarDataAnalysis(filename)
    % filename: .xlsx 数据文件名"

    %读取数据
    T = readtable(filename);

    %组合时间戳
    Time = T.Timestamp_Seconds + T.Timestamp_Nanoseconds * 1e-9;   % 单位：秒

    X = T.x_m;     % 米
    Y = T.y_m;
    Z = T.z_m;

    RangeRate = T.RangeRate_mS;   % m/s
    RCS       = T.RCS_dBm2;       % dBm^2

    %时间归一化
    cmap = jet(256);
    t_norm = (Time - min(Time)) / (max(Time) - min(Time));  
    colorIdx = max(1, round(t_norm * 255) + 1);

    %绘制散点
    figure('Color','w');
    scatter3(X, Y, Z, 36, cmap(colorIdx,:), 'filled');
    xlabel('X (m)', 'FontSize', 12);
    ylabel('Y (m)', 'FontSize', 12);
    zlabel('Z (m)', 'FontSize', 12);
    title('Radar Scatter Points Colored by Timestamp', 'FontSize',14);

    grid on;
    axis equal;

    colormap(cmap);
    cb = colorbar;
    cb.Label.String = 'Timestamp (old → new)';

    %点击散点时显示 RCS 和 RangeRate
    dcm = datacursormode(gcf);
    set(dcm, 'UpdateFcn', @(obj,event) pointInfoCallback(event, X, Y, Z, Time, RCS, RangeRate));
end

%光标回调
function txt = pointInfoCallback(event, X, Y, Z, Time, RCS, RangeRate)
    pos = event.Position;
    idx = find(X == pos(1) & Y == pos(2) & Z == pos(3), 1);

    txt = {
        sprintf('X = %.3f m', pos(1))
        sprintf('Y = %.3f m', pos(2))
        sprintf('Z = %.3f m', pos(3))
        sprintf('Time = %.9f s', Time(idx))   % 保留纳秒信息
        sprintf('RCS = %.2f dBm^2', RCS(idx))
        sprintf('RangeRate = %.3f m/s', RangeRate(idx))
    };
end

%%使用时在控制台输入 RadarDataAnalysis('要处理的文件地址和文件名.xlsx')