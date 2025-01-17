const convertLinks = ( input ) => {

    let text = input;
    const linksFound = text.match( /(?:www|https?)[^\s]+/g );
    const aLink = [];
  
    if ( linksFound != null ) {
  
      for ( let i=0; i<linksFound.length; i++ ) {
        let replace = linksFound[i];
        if ( !( linksFound[i].match( /(http(s?)):\/\// ) ) ) { replace = 'http://' + linksFound[i] }
        let linkText = replace.split( '/' )[2];
        if ( linkText.substring( 0, 3 ) == 'www' ) { linkText = linkText.replace( 'www.', '' ) }
        if ( linkText.match( /youtu/ ) ) {
  
          let youtubeID = replace.split( '/' ).slice(-1)[0];
          aLink.push( '<div class="video-wrapper"><iframe src="https://www.youtube.com/embed/' + youtubeID + '" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>' )
        }
        else if ( linkText.match( /vimeo/ ) ) {
          let vimeoID = replace.split( '/' ).slice(-1)[0];
          aLink.push( '<div class="video-wrapper"><iframe src="https://player.vimeo.com/video/' + vimeoID + '" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></div>' )
        }
        else {
          aLink.push( '<a href="' + replace + '" target="_blank">' + linkText + '</a>' );
        }
        text = text.split( linksFound[i] ).map(item => { return aLink[i].includes('iframe') ? item.trim() : item } ).join( aLink[i] );
      }
      return text;
  
    }
    else {
      return input;
    }
  }