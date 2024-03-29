function [filename,metadata,n_plotted]=optic_flow_dots(varargin)
    
    %OPTIC_FLOW_DOTS    Generate optic flow movie files.
    %
    %   OPTIC_FLOW_DOTS generates an optic flow movie file that simulates
    %   arbitrary translational and rotational motions of a camera through a
    %   cube-shaped environment of dots. The sides of the cube have arbitrary
    %   length 2. Only the dots within a sphere centered on the camera are
    %   visible. The diameter of the sphere is also 2. The purpose of this
    %   sphere is to keep the dot density constant in any direction.
    %
    %   As the camera moves, dots that cross a side of the cube are wrapped
    %   around. To prevent cyclical looping of the same dots, the wrapped dots
    %   are given fresh random coordinates for their non-wrapped dimensions.
    %   For example, if a dot drops out of the backside of the unit box (Z<-1),
    %   Z is incremented by 2 and X and Y are set to random values between -1
    %   and 1.
    %
    %   The translation and rotation parameters can be constant throughout the
    %   video, or dynamically change from frame to frame. See the descriptions
    %   of the 'trans_xyz', 'rot_xyz', and 'rot_dpf' parameters below for more
    %   information on this.
    %
    %   Other features include setting the background color; shape, size, and
    %   color of the dots; limited dot lifetime; toggle erasing between frames
    %   on and off; and setting the limits of the dot enviroment to create, for
    %   example, groundplanes or fronto-parallel planes of dots.
    %
    %   Settings are controlled using the following NAME,VALUE pair arguments
    %   (default values in brackets):
    %
    %       'wid_px': Width of the video in pixels. [500]
    %       'hei_px': Height of the video in pixels. [500]
    %       'back_rgb': Color of the background in red,green,blue values
    %           between 0 and 1 or a color character such as 'w' for white or
    %           ['k'] for black. (see: help plot).
    %       'n_frames': The number of video frames to be generated. [100]
    %       'n_ndots': The number of dots in the environment (the number of
    %           visible dots will be less than this). [500]
    %       'dot_style': Cell array with name,value pairs as the builtin
    %           function plot takes tem (see: help plot).
    %           [{'wo', 'MarkerSize',2, 'MarkerFaceColor','w'}]
    %       'erase': Clear the plot each frame, [true] or false.
    %       'dot_x_range': Keep all dots' X-coordinatess (left-right) fixed to
    %           the provide value between -1 and 1. Default, [], fills the
    %           entire dimension evenly with dots.
    %       'fix_dot_y': Same as fix_dot_x but for the up-down axes Y. Use this
    %           to simulate movement over a ground plane.
    %       'fix_dot_z': Same as fix_dot_z but for the up-down axes Y. Use this
    %           to simulate movement toward a fronto-parallel plane of dots.
    %       'trans_xyz': The camera translation per frame, [X;Y;Z], where
    %           positive X, Y, and Z give right, up, and forward motion,
    %           respectively, in arbitrary units of length per frame. Must be a
    %           3-element column vector or a 3 x n_frames matrix with a column
    %           vector with each frame. The larger the values, the higher the
    %           translation speed. Keep in mind that the entire dot environment
    %           has a radius of 1, so sensible values are typically (much)
    %           smaller than 1. [[0;0;0.05]]
    %       'rot_xyz': The camera rotation-axes. Format like trans_xyz, i.e. a
    %           3x1 vector or a 3 x n_frames matrix. The length of this vector
    %           is irrelevant, it's the scalar rot_dpf (see below) that
    %           determines the rotation rate, i.e., [0;1;0] and [0;100;0]
    %           equivalently specify a vertical axes that can be used for
    %           sideways rotation of the camera. [[0;1;0]]
    %       'rot_dpf': The camera rotation-rate in degrees per frame. Either a
    %           scalar for constant rotation or a 1 x n_frames vector with
    %           variable rotation rates. Rotation follows the "righthand rule",
    %           meaning that, for example, a positive rot_dpf around a vertical
    %           axis that points up ([0;1;0]) makes the camera pan to the left.
    %           [0]
    %       'dot_life_fr': The number of frames until a dot's position is
    %           reset to a random location in the enviroment. A value of 2
    %           means that the dots take a single step before being
    %           repositioned. A value of 1 means the stimulus is white noise. A
    %           value of [0] means the dots have unlimited life.
    %       'dot_life_sync': Refresh all dots at once or asynchronous.
    %           [false]
    %       'viewport': Rectangle [Xmin Xmax Ymin Ymax] that acts as a window
    %           at Z=0. Default, [sind([-45 45 -45 45])], cuts the largest
    %           possible square out of the Z=0 cross section of the sphere of
    %           visibility.
    %       'video_filename': The outputs video's filename. Leave empty to not
    %           save a video. Extension will be autogenerated depending on
    %           'video_profile'. ['optic_flow_dots_video']
    %       'video_profile': The video profile, for example 'MPEG-4'. Call
    %           <a href="matlab:VideoWriter.getProfiles">VideoWriter.getProfiles</a> for a list of profiles available
    %           on your computer with descriptions. ['Archival']
    %       'video_framerate': The frames per second of the video. [25]
    %
    %   filename = OPTIC_FLOW_DOTS(...) returns the name of the videofile
    %   (including the path).
    %
    %   [~,metadata] = OPTIC_FLOW_DOTS(...) returns all settings, the state of
    %   the random number generator as it was at the start of the simulation
    %   (to be able to exactly replicate it), and the date and start time of
    %   the simulation.
    %
    %   [~,~,n_plotted] = OPTIC_FLOW_DOTS(...) returns an 1 x n_frames vector
    %   of the number of dots plotted in each frame. The number of dots plotted
    %   likely exceeds the number of visible dots because of dot overlap.
    %
    %   I chose to make this function write to a video instead of outputting a
    %   4D matrix (width x height x 3(RGB) x n_frames) because saving the
    %   frames as a movie file on disk scales better to long and large field
    %   animations than keeping all frames in RAM. Use VideoReader to read the
    %   frames from the file one-by-one, or all at once, or in any desired
    %   subset. Note, however, that this is only equivalent to creating the
    %   entire stack in one go when no or lossless video compression is used.
    %   So if perfect fidelity is desired, use for example Archival (the
    %   default) as the video profile instead of a one with lossy compression
    %   such as MPEG-4.
    %
    %   Examples:
    %
    %       % Simulate backward motion through a cloud of large red stars while
    %       % rotating clockwise around the line of sight, and save it to a
    %       % file called 'test.mp4' in the current working directory.
    %       vidfilename = optic_flow_dots('trans_xyz',[0;0;-0.02], ...
    %           'rot_xyz',[0;0;1], 'rot_dpf',2, ...
    %           'dot_style',{'r*','MarkerSize',15}, ...
    %           'video_filename','test','video_profile','MPEG-4');
    %
    %       % Simulate forward motion over a plane of dots while rotating
    %       % at 0.5 degrees per frame around a vertical axis through the
    %       % camera, do not create a video file but do output the metadata
    %       % and the number of visible dots per frame
    %       [vidfilename, metadata, nvis] = optic_flow_dots( ...
    %           'trans_xyz',[0;0;0.01], ...
    %           'rot_xyz',[0;1;0],'rot_dpf',0.5, ...
    %           'y_range',[-0.05 -0.05], ...
    %           'video_filename',[])
    %
    %   Tips:
    %      - Use implay(vidfilename) to view the movie in matlab
    %      - Use V=VideoReader(vidfilename); all_frames=V.read; to convert the
    %        entire video to a matrix.
    %      - See help VideoReader for more ways to get the frames from the
    %        movie file.
    %
    %   See also: VideoWriter, VideoWriter.getProfiles, VideoReader,
    %   getframe
    
    %   BSD 3-Clause License Copyright (c) 2019, Jacob Duijnhouwer
    
    p=inputParser;
    is_natural=@(x)isnumeric(x) && isreal(x) && all(rem(x,1)==0) && all(x>0);
    is_color=@(x)((isnumeric(x) && numel(x)==3 && all(x>=0) && all(x<=1)) || any(x=='bgrcmykw'));
    is_boolean=@(x)(numel(x)==1 && (islogical(x) || x==1 || x==0));
    is_within_unit_range=@(x)numel(x)==2 && all(x>=-1 & x<=1) && x(1)<=x(2);
    p.addOptional('n_dots',500,@(x)is_natural(x) && numel(x)==1);
    p.addOptional('x_range',[-1 1],@(x)is_within_unit_range(x));
    p.addOptional('y_range',[-1 1],@(x)is_within_unit_range(x));
    p.addOptional('z_range',[-1 1],@(x)is_within_unit_range(x));
    p.addOptional('wid_px',500,@(x)is_natural(x) && numel(x)==1);
    p.addOptional('hei_px',500,@(x)is_natural(x) && numel(x)==1);
    p.addOptional('fig_pos_px',[],@(x)(is_natural(x) && numel(x)==4) || isempty(x));
    p.addOptional('back_rgb','k',@(x)is_color(x));
    p.addOptional('erase',true,@(x)is_boolean(x));
    p.addOptional('dot_style',{'wo','MarkerSize',2,'MarkerFaceColor','w'});
    p.addOptional('n_frames',100,@(x)is_natural(x) && numel(x)==1);
    p.addOptional('trans_xyz',[0;0;0.05],@(x)isnumeric(x) && isreal(x) && size(x,1)==3);
    p.addOptional('rot_xyz',[0;1;0],@(x)isnumeric(x) && isreal(x) && size(x,1)==3);
    p.addOptional('rot_dpf',0,@(x)isnumeric(x) && isreal(x));
    p.addOptional('dot_life_fr',0,@(x)is_natural(x) || x==0);
    p.addOptional('dot_life_sync',true,@(x)is_boolean(x));
    p.addOptional('viewport',sind([-45 45 -45 45]),@(x)isnumeric(x) && numel(x)==4 && x(1)<x(2) && x(3)<x(4));
    p.addOptional('video_filename','optic_flow_dots_video',@(x)ischar(x) || isempty(x));
    p.addOptional('video_profile','Archival',@ischar);
    p.addOptional('video_framerate',25,@isnumeric);
    p.parse(varargin{:});
    
    % Open the videowriter object. Do this dirst because this step has the
    % highest likelihood of failing, for example if a file with the same
    % name is still open somewhere.
    try
        if ~isempty(p.Results.video_filename)
            vid = VideoWriter(p.Results.video_filename,p.Results.video_profile);
            vid.FrameRate = p.Results.video_framerate;
            open(vid);
        end
    catch me
        disp('[optic_flow_dots] Could not open the videofile for writing. It may already be open somewhere. Try fclose(''all'') to close it.');
        rethrow(me);
    end
    
    % Create the metadata
    metadata=p.Results;
    metadata.start_rng_state=rng;
    metadata.start_date=datestr(clock,0);
    
    % check that the transformation is constant or dynamic per frame, check
    % for correct input format, too (3x1 or 3xN_frames matrices)
    err={};
    if ~any(size(p.Results.trans_xyz,2)==[1 p.Results.n_frames])
        err{end+1}='trans_xyz must be a 3 element column vector or a 3 x n_frames matrix';
    end
    if ~any(size(p.Results.rot_xyz,2)==[1 p.Results.n_frames])
        err{end+1}='rot_xyz must be a 3 element column vector or a 3 x n_frames matrix';
    end
    if ~any(numel(p.Results.rot_dpf)==[1 p.Results.n_frames])
        err{end+1}='rot_dpf must be a scalar or a vector with n_frames elements';
    end
    if ~isempty(err)
        error([mfilename ':wrongtransform'],'  %s\n',err{:});
    end
     
    % Create the figure window
    fig=figure('Name', mfilename,'Visible','off');
    fig.NumberTitle='off';
    fig.CloseRequestFcn=@(~,~,~)evalin('caller','figure_close_requested=true;');
    fig.Units='pixels';
    fig.MenuBar='none';
    fig.Renderer='OpenGL';
    fig.Color=p.Results.back_rgb;
    if isempty(p.Results.fig_pos_px)
        fig.Position=[120 120 max(p.Results.hei_px+1,300) max(p.Results.wid_px+1,100)];
        movegui(fig,'center');
    else
        fig.Position=p.Results.fig_pos_px;
    end
    fig.Resize='off';
    fig.Visible='on';
    
    
    % Create the axes and fix them
    ax=axes(fig);
    ax.Units='pixels';
    ax.Position=[1 1 p.Results.wid_px p.Results.hei_px];
    ax.Color=p.Results.back_rgb;
    ax.XLimMode='manual';
    ax.XLim=p.Results.viewport(1:2);
    ax.YLimMode='manual';
    ax.YLim=p.Results.viewport(3:4);
    hold(ax,'on')
    axis(ax,'off');
    
    % give OS time to position and size window perfectly
    for i=1:10
        fr=getframe(ax);
        if size(fr.cdata,1)~=p.Results.hei_px || size(fr.cdata,2)~=p.Results.wid_px
            pause(0.1);
            drawnow;
        else
            break;
        end
    end
    if i==10
        error('Video doesnt fit in window, increase fig_pos_px(3:4)');
    end
    
    
    % Create a fresh box with n_dots random dots
    dots_xyz=ones(4,p.Results.n_dots);
    dots_xyz(1,:)=rand(1,p.Results.n_dots)*diff(p.Results.x_range)+p.Results.x_range(1);
    dots_xyz(2,:)=rand(1,p.Results.n_dots)*diff(p.Results.y_range)+p.Results.y_range(1);
    dots_xyz(3,:)=rand(1,p.Results.n_dots)*diff(p.Results.z_range)+p.Results.z_range(1);
    
    % If the dots are to have limited lifetime, set there remaining frames now
    if p.Results.dot_life_fr>0 && ~p.Results.dot_life_sync
        dots_frames_left=randi(p.Results.dot_life_fr,1,p.Results.n_dots);
    end
    
    % Allocate the list of visible dots per frame, if that output is requested
    if nargout>2
        n_plotted=nan(1,p.Results.n_frames);
    end
    
    % Draw until all frames are completed or until user closes window
    figure_close_requested=false;
    for fr=1:p.Results.n_frames
        % stop if the user closes the window
        if figure_close_requested
            break; %#ok<UNRCH>
        end
        
        % Update the name of the figure to represent progress
        if round((fr-1)/p.Results.n_frames*100) < round(fr/p.Results.n_frames*100) || fr==1
            fig.Name=sprintf('Recording: %s (%d%%)',vid.Filename,round(fr/p.Results.n_frames*100));
        end
        
        % Make the transformation matrix M
        T=p.Results.trans_xyz(:,mod(fr-1,size(p.Results.trans_xyz,2))+1);
        R=p.Results.rot_xyz(:,mod(fr-1,size(p.Results.rot_xyz,2))+1);
        A=p.Results.rot_dpf(mod(fr-1,numel(p.Results.rot_dpf))+1);
        M=makehgtform('axisrotate',R,A/180*pi,'translate',-T);
        
        % Apply the transformation to the environment
        dots_xyz=M*dots_xyz;
        
        % Refresh dots that have reached the end of their lifetime
        if p.Results.dot_life_fr>0
            if p.Results.dot_life_sync
                if mod(fr-1,p.Results.dot_life_fr)==0
                    % reposition all dots at once
                    dots_xyz(1:3,:)=rand(3,size(dots_xyz,2))*2-1; 
                end
            else
                dots_frames_left=dots_frames_left-1;
                expired=dots_frames_left==0;
                % Give new random positions to the expired dots
                dots_xyz(1:3,expired)=rand(3,sum(expired))*2-1;
                % Refresh their frames_left counter to max life time
                dots_frames_left(expired)=p.Results.dot_life_fr;
            end
        end
        
        % Detect dot coordinates that need wrapping (outside unit box)
        too_neg=dots_xyz<-1;
        too_pos=dots_xyz>1;
        % Get new random coordinates for all dots, store current in
        % prewrap_xyz
        prewrap_xyz = dots_xyz;
        dots_xyz = ones(4,p.Results.n_dots);
        dots_xyz(1,:) = rand(1,p.Results.n_dots)*diff(p.Results.x_range)+p.Results.x_range(1);
        dots_xyz(2,:) = rand(1,p.Results.n_dots)*diff(p.Results.y_range)+p.Results.y_range(1);
        dots_xyz(3,:) = rand(1,p.Results.n_dots)*diff(p.Results.z_range)+p.Results.z_range(1);
        % Put all XYZ-s back of the dots that were fine (within the box)
        % and, of the dots that were not fine, only the offending
        % coordinates (so that they can be wrapped to the other side of the
        % box in the next step)
        fine = repmat(~any(too_neg|too_pos,1),4,1);
        dots_xyz(fine|too_neg|too_pos) = prewrap_xyz(fine|too_neg|too_pos);
        % Wrap the offending coordinates so that they are fine now
        dots_xyz(too_neg) = dots_xyz(too_neg)+2;
        dots_xyz(too_pos) = dots_xyz(too_pos)-2;
        
        % Detect which dots are in unit sphere and in front of camera
        in_frontal_half_dome = vecnorm(dots_xyz(1:3,:))<=1 & dots_xyz(3,:)>0;
        X = dots_xyz(1,in_frontal_half_dome)./(dots_xyz(3,in_frontal_half_dome));
        Y = dots_xyz(2,in_frontal_half_dome)./(dots_xyz(3,in_frontal_half_dome));
        
        % Append the number of visible stars to the list, if requested
        if nargin>2
            visible = inpolygon(X,Y,[ax.XLim ax.XLim(2) ax.XLim(1)],[ax.YLim(1) ax.YLim(1) ax.YLim(2) ax.YLim(2)]);
            n_plotted(fr)=numel(visible);
        end
        
        % Clear the axes and plot the dots
        if p.Results.erase
            cla(ax)
        else
            hold(ax,'on');
        end
        plot(ax,X,Y,p.Results.dot_style{:});
        
        % flush the stack
        drawnow;
        
        % wait a little bit to allow matlab to detect if the user pressed
        % on the closure cross in the top-right corner of the figure window
        pause(0.001);
        
        % If requested, append a frame to the video
        if ~isempty(p.Results.video_filename)
            thisframe=getframe(ax);
            if strcmp(vid.VideoFormat,'Grayscale Avi')
                thisframe.cdata=mean(double(thisframe.cdata),3)/255;
            end
            writeVideo(vid,thisframe);
        end
        
    end
    
    % close the window
    delete(fig);
    
    % close the video file and create output 'filename'
    if exist('vid','var') % doesn not exist if filename was empty
        filename=fullfile(vid.Path,vid.Filename);
        metadata.video_filename=filename;
        close(vid);
    else
        filename='';
    end
    
    function error(varargin)
        warning off
        close(vid);
        warning on
        if exist('fig','var')
            delete(fig);
        end
        builtin('error',varargin{:});
    end
end



